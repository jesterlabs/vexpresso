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
        data: Optional[Union[str, pd.DataFrame]] = None,
        retriever: Retriever = NumpyRetriever(),
        embedding_functions: Dict[str, Any] = {},
        daft_df: Optional[daft.DataFrame] = None,
    ):
        self.df = daft_df
        self.retriever = retriever
        self.embedding_functions = embedding_functions

        _metadata_dict = {}

        if data is not None:
            if isinstance(data, str):
                if data.endswith(".json"):
                    with open(data, "r") as f:
                        data = pd.DataFrame(json.load(f))
            _metadata_dict = data.to_dict("list")

        if daft_df is None:
            self.df = daft.from_pydict({**_metadata_dict})
            self.df = self.df.with_column(
                "vexpresso_index", indices(col(self.column_names[0]))
            )

    def __len__(self) -> int:
        return self.df.count_rows()

    def __getitem__(self, column: str) -> Collection:
        return self.select(column)

    def __setitem__(self, column: str, value: List[Any]) -> None:
        self.df = self.add_column(column = value, name = column).df

    @property
    def column_names(self) -> List[str]:
        return self.df.column_names

    def from_df(self, df: daft.DataFrame) -> DaftCollection:
        return DaftCollection(
            retriever=self.retriever,
            embedding_functions = self.embedding_functions,
            daft_df=df,
        )

    def add_column(self, column: List[Any], name: str = None) -> DaftCollection:
        if name is None:
            num_columns = len(self.df.column_names)
            name = f"column_{num_columns}"

        new_df = daft.from_pydict({name:column})
        df = self.df.with_column(name, new_df[name])
        return self.from_df(df)

    def col(self, *args) -> Scope:
        return Scope(columns=args, collection=self)

    def collect(self, in_place: bool = False):
        if in_place:
            self.df = self.df.collect()
            return self
        return self.from_df(self.df.collect())

    def execute(self) -> DaftCollection:
        return self.collect()

    @classmethod
    def from_collection(cls, collection: DaftCollection, **kwargs) -> DaftCollection:
        kwargs = {
            "daft_df": collection.df,
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
    ) -> daft.DataFrame:
        if query_embeddings is None:
            query_embeddings = self.embedding_fn.func(query, **embedding_fn_kwargs)

        embedding_column_name = content_name
        if embedding_column_name not in df.column_names:
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
        query:  Dict[str, List[Any]] = {},
        query_embeddings: Dict[str, List[Any]] = {},
        filter_conditions: Optional[Dict[str, Dict[str, str]]] = None,
        k=10,
        embedding_fn_kwargs = {}
    ) -> DaftCollection:
        df = self.df

        for key in query_embeddings:
            df = self._retrieve(
                df=df,
                content_name=key,
                query_embeddings=query_embeddings.get(key),
                k=k,
                embedding_fn_kwargs = embedding_fn_kwargs
            )

        for key in query:
            df = self._retrieve(
                df=df,
                content_name=key,
                query=query.get(key),
                k=k,
                embedding_fn_kwargs = embedding_fn_kwargs
            )

        if filter_conditions is not None:
            df = FilterHelper.filter(df, filter_conditions)

        return self.from_df(df)

    @lazy(default=True)
    def select(
        self,
        *args,
    ) -> DaftCollection:
        return self.from_df(FilterHelper.select(self.df, *args))

    @lazy(default=True)
    def filter(
        self, filter_conditions: Dict[str, Dict[str, str]], *args, **kwargs
    ) -> DaftCollection:
        return self.from_df(
            FilterHelper.filter(
                self.df,
                filter_conditions,
                *args,
                **kwargs
            )
        )

    @lazy(default=True)
    def apply(
        self,
        transform_fn: Transformation,
        *args,
        to: Optional[str] = None,
        **kwargs
    ) -> DaftCollection:
        if getattr(transform_fn, "__vexpresso_transform", None) is None:
            transform_fn = transformation(transform_fn)

        if not isinstance(_args[0], DaftCollection):
            raise TypeError("first args in apply must be a DaftCollection! use `collection['column_name']`")

        _args = []
        for _arg in args:
            if isinstance(_arg, DaftCollection):
                column = _arg.df.columns[0]
                _args.append(column)
            else:
                _args.append(_arg)

        _kwargs = {}
        for k in kwargs:
            _kwargs[k] = kwargs[k]
            if isinstance(_kwargs[k] , DaftCollection):
                # only support first column
                column = _kwargs[k].df.columns[0]
                _kwargs[k] = column

        if to is None:
            to = _args[0].name()

        df = df.with_column(to, transform_fn(*_args, **_kwargs))
        return self.from_df(df)

    @lazy(default=True)
    def embed(
        self,
        content: Optional[List[Any]] = None,
        column_name: Optional[str] = None,
        embedding_fn: Optional[Transformation] = None,
        update_embedding_fn: bool = True,
        *args,
        **kwargs
    ) -> DaftCollection:
        collection = self

        if content is None and column_name is None:
            raise ValueError("column_name or content must be specified!")

        if content is not None:
            collection = self.add_column(content, column_name)

        if embedding_fn is None:
            embedding_fn = self.embedding_functions[column_name]
        else:
            if column_name in embedding_fn:
                if embedding_fn != self.embedding_functions[column_name]:
                    print("embedding_fn may not be the same as whats in map!")
                if update_embedding_fn:
                    self.embedding_functions[column_name] = embedding_fn

        if getattr(self.embedding_functions[column_name], "__vexpresso_transform", None) is None:
            self.embedding_functions[column_name] = transformation(self.embedding_functions[column_name])

        kwargs = {
            "to": f"embeddings_{column_name}",
            **kwargs,
        }

        args = [self.collection[column_name], *args]

        return collection.apply(
            *args,
            **kwargs,
            transform_fn=self.embedding_functions[column_name],
        )

    def save_local(self, directory: str) -> str:
        os.makedirs(directory, exist_ok=True)
        table = self.df.to_arrow()
        pq.write_table(table, os.path.join(directory, "content.parquet"))

    @classmethod
    def from_local_dir(self, local_dir: str, *args, **kwargs) -> DaftCollection:
        df = daft.read_parquet(os.path.join(local_dir, "content.parquet"))
        return DaftCollection(df=df, *args, **kwargs)
