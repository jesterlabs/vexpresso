from __future__ import annotations

import json
import os
from functools import partial
from typing import Any, Callable, Dict, Iterable, List, Optional, Union

import daft
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from daft import col

from vexpresso.collection.collection import Collection, Plan
from vexpresso.collection.daft.filter import FilterHelper
from vexpresso.retriever import NumpyRetriever, Retriever
from vexpresso.transformation import CallableTransformation, Transformation  # noqa
from vexpresso.utils import deep_get


@daft.udf(return_dtype=daft.DataType.int64())
def indices(columnn):
    return list(range(len(columnn)))


def embed(column, embedding_fn):
    # dumb langchain check, might need something more specific here
    if getattr(embedding_fn, "embed_documents", None) is not None:
        return np.array(embedding_fn.embed_documents(column))
    return embedding_fn(column)


@daft.udf(return_dtype=daft.DataType.python())
def _retrieve(embedding_col, query_embeddings, retriever, k):
    embeddings = embedding_col.to_pylist()
    retrieval_output = retriever.retrieve(
        query_embeddings=query_embeddings, embeddings=embeddings, k=k
    )[0]
    indices = retrieval_output.indices
    scores = retrieval_output.scores

    results = [
        {"retrieve_index": None, "retrieve_score": scores[i]}
        for i in range(len(embeddings))
    ]

    for idx in indices:
        results[idx]["retrieve_index"] = idx

    return results


class DaftCollection(Collection):
    def __init__(
        self,
        content: Optional[Dict[str, Iterable[Any]]] = None,
        df: Optional[daft.DataFrame] = None,
        metadata: Optional[Union[str, pd.DataFrame]] = None,
        embedding_fn: Callable[[List[Any], List[Any]]] = None,
        retriever: Retriever = NumpyRetriever(),
        plan: List[Plan] = [],
        lazy_start: bool = False,
    ):
        super().__init__(plan, lazy_start)
        self.df = df
        self.embedding_fn = embedding_fn
        self.retriever = retriever

        _metadata_dict = {}

        if metadata is not None:
            if isinstance(metadata, str):
                if metadata.endswith(".json"):
                    with open(metadata, "r") as f:
                        metadata = pd.DataFrame(json.load(f))
            _metadata_dict = metadata.to_dict("list")

        if df is None:
            if isinstance(content, str):
                content = {f"{content}": deep_get(_metadata_dict, keys=content)}
            self.df = daft.from_pydict({**content, **_metadata_dict})

            columns = list(content.keys())
            columns = {
                f"embeddings_{c}": c
                for c in columns
                if f"embeddings_{c}" not in self.column_names
            }

            if len(columns) > 0 and self.embedding_fn is not None:
                collection = self.embed(
                    **columns,
                    lazy=lazy_start,
                    embedding_fn=self.embedding_fn,
                )
                self.df = collection.df
                self.plan = collection.plan

            self.df = self.df.with_column(
                "modal_map_index", indices(col(self.df.column_names[0]))
            )

    def _from_plan(self, plan: List[Plan]) -> DaftCollection:
        return DaftCollection(
            df=self.df,
            embedding_fn=self.embedding_fn,
            retriever=self.retriever,
            plan=plan,
        )

    def to_pandas(self) -> pd.DataFrame:
        collection = self.execute()
        return collection.df.to_pandas()

    def to_dict(self) -> Dict[str, List[Any]]:
        collection = self.execute()
        return collection.df.to_pydict()

    def show(self, num_rows: int):
        return self.df.show(num_rows)

    @property
    def indices(self) -> List[int]:
        return self.df.select("modal_map_index").to_pydict()["modal_map_index"]

    @property
    def column_names(self) -> List[str]:
        return self.df.column_names

    def embed(
        self, *args, lazy: bool = True, embedding_fn=None, **kwargs
    ) -> Collection:
        if embedding_fn is None:
            embedding_fn = self.embedding_fn
        transformation = partial(embed, embedding_fn=embedding_fn)
        if not isinstance(transformation, Transformation):
            transformation = CallableTransformation(transformation)
        return self.transform(*args, transform=transformation, lazy=lazy, **kwargs)

    def collect(self):
        return DaftCollection(
            df=self.df.collect(),
            embedding_fn=self.embedding_fn,
            retriever=self.retriever,
        )

    def retrieve(
        self,
        content_name: str,
        query: Union[str, List[Any]],
        query_embeddings=None,
        k: int = None,
        df=None,
    ):
        if df is None:
            df = self.df
        if query_embeddings is None:
            query_embeddings = embed(query, self.embedding_fn)

        embedding_column_name = f"embeddings_{content_name}"
        if embedding_column_name not in self.column_names:
            print(
                f"{embedding_column_name} not found in daft df. Embedding column {content_name}..."
            )
            df = self.embed(content_name)

        df = df.with_column(
            "retrieve_output",
            _retrieve(
                col(embedding_column_name),
                query_embeddings=query_embeddings,
                k=k,
                retriever=self.retriever,
            ),
        )
        df = (
            df.with_column(
                "retrieve_index",
                col("retrieve_output").apply(
                    lambda x: x["retrieve_index"], return_dtype=daft.DataType.int64()
                ),
            )
            .with_column(
                "score",
                col("retrieve_output").apply(
                    lambda x: x["retrieve_score"], return_dtype=daft.DataType.float64()
                ),
            )
            .exclude("retrieve_output")
            .where(col("retrieve_index") != -1)
            .exclude("retrieve_index")
            .sort(col("score"), desc=True)
        )

        return df

    def execute_query(
        self,
        query: Dict[str, List[Any]] = {},
        query_embeddings: Dict[str, Any] = {},
        k=10,
        *args,
        **kwargs,
    ) -> Collection:
        df = self.df
        for key in query:
            df = self.retrieve(
                content_name=key,
                query=query.get(key, None),
                query_embeddings=query_embeddings.get(key, None),
                k=k,
                df=df,
                *args,
                **kwargs,
            )

        return DaftCollection(
            df=df,
            embedding_fn=self.embedding_fn,
            retriever=self.retriever,
            plan=self.plan,
        )

    def execute_select(
        self,
        *args,
    ) -> DaftCollection:
        return DaftCollection(
            df=FilterHelper.select(self.df, *args),
            embedding_fn=self.embedding_fn,
            retriever=self.retriever,
            plan=self.plan,
        )

    def execute_filter(
        self, filter_conditions: Dict[str, Dict[str, str]], *args, **kwargs
    ) -> Collection:
        return DaftCollection(
            df=FilterHelper.filter(self.df, filter_conditions, *args, **kwargs),
            embedding_fn=self.embedding_fn,
            retriever=self.retriever,
            plan=self.plan,
        )

    def execute_transform(
        self,
        *args,
        transform: Union[Callable[[List[Any]], List[Any]], Transformation],
        **kwargs,
    ) -> DaftCollection:
        if not isinstance(transform, Transformation):
            # callable
            transform = CallableTransformation(transform)

        @daft.udf(return_dtype=daft.DataType.python())
        def _udf(column):
            column = column.to_pylist()
            return transform(column)

        df = self.df
        for c in args:
            df = df.with_column(f"transformed_{c}", _udf(col(c)))
        for k in kwargs:
            df = df.with_column(k, _udf(col(kwargs[k])))

        return DaftCollection(
            df=df,
            embedding_fn=self.embedding_fn,
            retriever=self.retriever,
            plan=self.plan,
        )

    def save_local(self, directory: str) -> str:
        os.makedirs(directory, exist_ok=True)
        table = self.df.to_arrow()
        pq.write_table(table, os.path.join(directory, "content.parquet"))

    @classmethod
    def from_local_dir(self, local_dir: str, *args, **kwargs) -> DaftCollection:
        df = daft.read_parquet(os.path.join(local_dir, "content.parquet"))
        return DaftCollection(df=df, *args, **kwargs)
