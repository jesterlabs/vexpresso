from __future__ import annotations

import json
import os
from typing import Any, Dict, Iterable, List, Optional, Union

import daft
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from daft import col

from vexpresso.collection.collection import Collection, Plan
from vexpresso.collection.daft.filter import FilterHelper
from vexpresso.collection.daft.utils import Transformation
from vexpresso.retriever import NumpyRetriever, Retriever
from vexpresso.utils import Column, deep_get


@daft.udf(return_dtype=daft.DataType.int64())
def indices(columnn):
    return list(range(len(columnn)))


def embed(content_list, embedding_fn):
    # dumb langchain check, might need something more specific here
    if getattr(embedding_fn, "embed_documents", None) is not None:
        return np.array(embedding_fn.embed_documents(content_list))
    return np.array(embedding_fn(content_list))


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
        embedding_fn: Transformation = None,
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
            content_dict = content
            if content is None:
                content_dict = {}
            elif isinstance(content, str):
                content_dict = {f"{content}": deep_get(_metadata_dict, keys=content)}
            self.df = daft.from_pydict({**content_dict, **_metadata_dict})

            columns = list(content_dict.keys())

            # this logic is a bit messy, probably need to clean it up
            if len(columns) > 0 and self.embedding_fn is not None:
                collection = self
                for c in columns:
                    collection = collection.embed(
                        column_name=c,
                        lazy=lazy_start,
                        embedding_fn=self.embedding_fn,
                        return_plan=False,
                    )
                self.df = collection.df
                self.plan = collection.plan

            self.df = self.df.with_column(
                "vexpresso_index", indices(col(self.df.column_names[0]))
            )

            if not lazy_start:
                self.df.collect()

    @classmethod
    def col(cls, name: str) -> Column:
        return Column(name)

    @classmethod
    def from_collection(cls, collection: DaftCollection, **kwargs) -> DaftCollection:
        kwargs = {
            "df": collection.df,
            "embeddings_fn": collection.embeddings_fn,
            "retriever": collection.retriever,
            "plan": collection.plan,
            **kwargs,
        }
        return DaftCollection(**kwargs)

    def clone(self, **kwargs) -> DaftCollection:
        kwargs = {
            "df": self.df,
            "embeddings_fn": self.embeddings_fn,
            "retriever": self.retriever,
            "plan": self.plan,
            **kwargs,
        }
        return DaftCollection(**kwargs)

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

    def to_list(self) -> List[Any]:
        collection = self.execute()
        return list(collection.df.to_pydict().values())

    def show(self, num_rows: int):
        return self.df.show(num_rows)

    @property
    def indices(self) -> List[int]:
        return self.df.select("vexpresso_index").to_pydict()["vexpresso_index"]

    @property
    def column_names(self) -> List[str]:
        return self.df.column_names

    def embed(
        self,
        *args,
        column_name: str,
        lazy: bool = True,
        embedding_fn: Transformation = None,
        return_plan: bool = False,
        **kwargs,
    ) -> Collection:
        # reset embedding_fn
        self.embedding_fn = embedding_fn
        args = [Column(column_name), *args]

        kwargs = {
            "to": f"embeddings_{column_name}",
            "transform": self.embedding_fn,
            "lazy": lazy,
            "return_plan": return_plan,
            **kwargs,
        }
        return self.transform(
            *args,
            **kwargs,
        )

    def collect(self, in_place: bool = False):
        if in_place:
            self.df = self.df.collect()
            self.plan = []
            return self
        return DaftCollection(
            df=self.df.collect(),
            embedding_fn=self.embedding_fn,
            retriever=self.retriever,
        )

    def retrieve(
        self,
        df,
        content_name: str,
        query: Union[str, List[Any]],
        query_embeddings=None,
        k: int = None,
        *args,
        **kwargs,
    ):
        if query_embeddings is None:
            query_embeddings = self.embedding_fn.func(query, *args, **kwargs)

        embedding_column_name = f"embeddings_{content_name}"
        if embedding_column_name not in self.column_names:
            raise ValueError(
                f"{embedding_column_name} not found in daft df. Make sure to call `embed` on column {content_name}..."
            )

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
                df=df,
                content_name=key,
                query=query.get(key, None),
                query_embeddings=query_embeddings.get(key, None),
                k=k,
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
        to: str,
        transform: Transformation,
        **kwargs,
    ) -> DaftCollection:
        destination = to

        _udf = daft.udf(return_dtype=daft.DataType.python())(transform)

        _args = []
        for c in args:
            if isinstance(c, Column):
                _args.append(col(c.name))
            else:
                _args.append(c)

        _kwargs = {}
        for k in kwargs:
            if isinstance(kwargs[k], Column):
                _kwargs[k] = col(kwargs[k])
            else:
                _kwargs[k] = kwargs[k]

        df = self.df.with_column(destination, _udf(*_args, **_kwargs))

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
