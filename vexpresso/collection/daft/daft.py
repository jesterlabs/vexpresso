from __future__ import annotations

import json
import os
from typing import Any, Dict, Iterable, List, Optional, Union

import daft
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from daft import col

from vexpresso.collection.collection import Collection
from vexpresso.collection.daft.filter import FilterHelper
from vexpresso.collection.daft.utils import Scope, Transformation, lazy, transformation
from vexpresso.retriever import NumpyRetriever, Retriever
from vexpresso.utils import deep_get


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
        lazy_start: bool = False,
    ):
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
                collection = self.col(*columns).embed(self.embedding_fn).collection
                self.df = collection.df

            self.df = self.df.with_column(
                "vexpresso_index", indices(col(self.df.column_names[0]))
            )

            if not lazy_start:
                self.df = self.df.collect()

    @property
    def column_names(self) -> List[str]:
        return self.df.column_names

    def col(self, *args) -> Scope:
        return Scope(columns=args, collection=self)

    def collect(self, in_place: bool = False):
        if in_place:
            self.df = self.df.collect()
            return self
        return DaftCollection(
            df=self.df.collect(),
            embedding_fn=self.embedding_fn,
            retriever=self.retriever,
        )

    def execute(self) -> DaftCollection:
        return self.collect()

    def from_df(self, df: daft.DataFrame):
        return DaftCollection(
            df=df,
            embedding_fn=self.embedding_fn,
            retriever=self.retriever,
        )

    @classmethod
    def from_collection(cls, collection: DaftCollection, **kwargs) -> DaftCollection:
        kwargs = {
            "df": collection.df,
            "embeddings_fn": collection.embeddings_fn,
            "retriever": collection.retriever,
            **kwargs,
        }
        return DaftCollection(**kwargs)

    def clone(self, **kwargs) -> DaftCollection:
        kwargs = {
            "df": self.df,
            "embeddings_fn": self.embeddings_fn,
            "retriever": self.retriever,
            **kwargs,
        }
        return DaftCollection(**kwargs)

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

    def _retrieve(
        self,
        df,
        content_name: str,
        query: Union[str, List[Any]],
        query_embeddings=None,
        k: int = None,
        embedding_fn_kwargs = {}
    ):
        if query_embeddings is None:
            query_embeddings = self.embedding_fn.func(query, **embedding_fn_kwargs)

        embedding_column_name = content_name
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

    @lazy(default=True)
    def query(
        self,
        query: Dict[str, List[Any]] = {},
        query_embeddings: Dict[str, Any] = {},
        filter_conditions: Optional[Dict[str, Dict[str, str]]] = None,
        k=10,
        embedding_fn_kwargs = {}
    ) -> Collection:
        df = self.df
        for key in query:
            df = self._retrieve(
                df=df,
                content_name=key,
                query=query.get(key, None),
                query_embeddings=query_embeddings.get(key, None),
                k=k,
                embedding_fn_kwargs = embedding_fn_kwargs
            )

        if filter_conditions is not None:
            df = FilterHelper.filter(df, filter_conditions)

        return DaftCollection(
            df=df,
            embedding_fn=self.embedding_fn,
            retriever=self.retriever,
        )

    @lazy(default=True)
    def select(
        self,
        *args,
    ) -> DaftCollection:
        return DaftCollection(
            df=FilterHelper.select(self.df, *args),
            embedding_fn=self.embedding_fn,
            retriever=self.retriever,
        )

    @lazy(default=True)
    def filter(
        self, filter_conditions: Dict[str, Dict[str, str]], *args, **kwargs
    ) -> Collection:
        return DaftCollection(
            df=FilterHelper.filter(self.df, filter_conditions, *args, **kwargs),
            embedding_fn=self.embedding_fn,
            retriever=self.retriever,
        )

    @lazy(default=True)
    def apply(
        self,
        columns: List[str],
        transform_fn: Transformation,
        to: str,
        fn_kwargs = {},
        map_columns: bool = True
    ) -> DaftCollection:
        destination = to

        inp = [col(c) for c in columns]
        if not map_columns:
            inp = [inp]

        for _args in inp:
            _kwargs = {}
            for k in fn_kwargs:
                kwarg = fn_kwargs[k]
                if isinstance(kwarg, str):
                    if kwarg.startswith("column."):
                        kwarg = col(kwarg.split("column.")[-1])
                _kwargs[k] = kwarg

            if getattr(transform_fn, "__vexpresso_transform", None) is None:
                transform_fn = transformation(transform_fn)

        df = self.df.with_column(destination, transform_fn(*_args, **_kwargs))

        return DaftCollection(
            df=df,
            embedding_fn=self.embedding_fn,
            retriever=self.retriever,
        )

    @lazy(default=True)
    def embed(
        self,
        column: str,
        embedding_fn: Transformation = None,
        *args,
        **kwargs,
    ) -> DaftCollection:
        # reset embedding_fn
        if embedding_fn is None:
            embedding_fn = self.embedding_fn
        self.embedding_fn = embedding_fn

        kwargs = {
            "to": f"embeddings_{column}",
            **kwargs,
        }

        return self.apply(
            column=column,
            transform_fn=self.embedding_fn,
            *args,
            **kwargs,
        )

    def save_local(self, directory: str) -> str:
        os.makedirs(directory, exist_ok=True)
        table = self.df.to_arrow()
        pq.write_table(table, os.path.join(directory, "content.parquet"))

    @classmethod
    def from_local_dir(self, local_dir: str, *args, **kwargs) -> DaftCollection:
        df = daft.read_parquet(os.path.join(local_dir, "content.parquet"))
        return DaftCollection(df=df, *args, **kwargs)
