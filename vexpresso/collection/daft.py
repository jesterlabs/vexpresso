from __future__ import annotations

import inspect
import os
from functools import reduce
from typing import Any, Dict, Iterable, List, Optional, Union

import daft
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from daft import col
from daft.datatype import DataType
from daft.expressions import Expression

from vexpresso.collection.collection import Collection, Plan
from vexpresso.retriever import NumpyRetriever, Retriever
from vexpresso.utils import deep_get


class FilterMethods:
    @classmethod
    def filter_methods(cls) -> Dict[str, Any]:
        NON_FILTER_METHODS = ["filter_methods", "print_filter_methods"]
        methods = {
            m[0]: m[1]
            for m in inspect.getmembers(cls)
            if not m[0].startswith("_") and m[0] not in NON_FILTER_METHODS
        }
        return methods

    @classmethod
    def print_filter_methods(cls):
        filter_methods = cls.filter_methods()
        for method in filter_methods:
            description = filter_methods[method].__doc__
            print(f"{method}: {description}")
            print("----------------------------------")

    @classmethod
    def eq(cls, field: str, value: Union[str, int, float]) -> Expression:
        """
        {field} equal to {value} (str, int, float)
        """

        field_name = field.split(".")[0]

        def _apply_fn(col_val) -> bool:
            keys = field
            if "." in field:
                keys = field.split(".", 1)[-1]

            return deep_get(col_val, keys=keys) == value

        return col(field_name).apply(_apply_fn, return_dtype=DataType.bool())

    @classmethod
    def neq(cls, field: str, value: Union[str, int, float]) -> Expression:
        """
        {field} not equal to {value} (str, int, float)
        """
        field_name = field.split(".")[0]

        def _apply_fn(col_val) -> bool:
            keys = field
            if "." in field:
                keys = field.split(".", 1)[-1]

            return deep_get(col_val, keys=keys) != value

        return col(field_name).apply(_apply_fn, return_dtype=DataType.bool())

    @classmethod
    def gt(cls, field: str, value: Union[int, float]) -> Expression:
        """
        {field} greater than {value} (int, float)
        """
        field_name = field.split(".")[0]

        def _apply_fn(col_val) -> bool:
            keys = field
            if "." in field:
                keys = field.split(".", 1)[-1]

            return deep_get(col_val, keys=keys) > value

        return col(field_name).apply(_apply_fn, return_dtype=DataType.bool())

    @classmethod
    def gte(cls, field: str, value: Union[int, float]) -> Expression:
        """
        {field} greater than or equal to {value} (int, float)
        """
        field_name = field.split(".")[0]

        def _apply_fn(col_val) -> bool:
            keys = field
            if "." in field:
                keys = field.split(".", 1)[-1]

            return deep_get(col_val, keys=keys) >= value

        return col(field_name).apply(_apply_fn, return_dtype=DataType.bool())

    @classmethod
    def lt(cls, field: str, value: Union[int, float]) -> Expression:
        """
        {field} less than {value} (int, float)
        """
        field_name = field.split(".")[0]

        def _apply_fn(col_val) -> bool:
            keys = field
            if "." in field:
                keys = field.split(".", 1)[-1]

            return deep_get(col_val, keys=keys) < value

        return col(field_name).apply(_apply_fn, return_dtype=DataType.bool())

    @classmethod
    def lte(cls, field: str, value: Union[int, float]) -> Expression:
        """
        {field} less than or equal to {value} (int, float)
        """
        field_name = field.split(".")[0]

        def _apply_fn(col_val) -> bool:
            keys = field
            if "." in field:
                keys = field.split(".", 1)[-1]

            return deep_get(col_val, keys=keys) <= value

        return col(field_name).apply(_apply_fn, return_dtype=DataType.bool())

    @classmethod
    def isin(cls, field: str, values: List[Union[str, int, float]]) -> Expression:
        """
        {field} is in list of {values} (list of str, int, or float)
        """
        field_name = field.split(".")[0]

        def _apply_fn(col_val) -> bool:
            keys = field
            if "." in field:
                keys = field.split(".", 1)[-1]

            return deep_get(col_val, keys=keys) in values

        return col(field_name).apply(_apply_fn, return_dtype=DataType.bool())

    @classmethod
    def notin(cls, field: str, values: List[Union[str, int, float]]) -> Expression:
        """
        {field} not in list of {values} (list of str, int, or float)
        """
        field_name = field.split(".")[0]

        def _apply_fn(col_val) -> bool:
            keys = field
            if "." in field:
                keys = field.split(".", 1)[-1]

            return deep_get(col_val, keys=keys) not in values

        return col(field_name).apply(_apply_fn, return_dtype=DataType.bool())

    @classmethod
    def contains(cls, field: str, value: Union[str, int, float]) -> Expression:
        """
        {field} (str) contains {value} (str)
        """
        field_name = field.split(".")[0]

        def _apply_fn(col_val) -> bool:
            keys = field
            if "." in field:
                keys = field.split(".", 1)[-1]

            return value in deep_get(col_val, keys=keys)

        return col(field_name).apply(_apply_fn, return_dtype=DataType.bool())

    @classmethod
    def notcontains(cls, field: str, value: Union[str, int, float]) -> Expression:
        """
        {field} (str) does not contains {value} (str)
        """
        field_name = field.split(".")[0]

        def _apply_fn(col_val) -> bool:
            keys = field
            if "." in field:
                keys = field.split(".", 1)[-1]

            return value not in deep_get(col_val, keys=keys)

        return col(field_name).apply(_apply_fn, return_dtype=DataType.bool())


class FilterHelper:
    FILTER_METHODS = FilterMethods.filter_methods()

    @classmethod
    def filter(
        cls, df: daft.DataFrame, filter_conditions: Dict[str, Dict[str, str]]
    ) -> daft.DataFrame:
        filters = []
        for metadata_field in filter_conditions:
            metadata_conditions = filter_conditions[metadata_field]
            for filter_method in metadata_conditions:
                if filter_method not in cls.FILTER_METHODS:
                    raise ValueError(
                        f"""
                            filter_method: {filter_method} not in supported filter methods: {cls.FILTER_METHODS}.
                        """
                    )
                value = metadata_conditions[filter_method]
                filters.append(cls.FILTER_METHODS[filter_method](metadata_field, value))
        op: Expression = reduce(lambda a, b: a & b, filters)
        return df.where(op)


@daft.udf(return_dtype=daft.DataType.int64())
def indices(columnn):
    return list(range(len(columnn)))


@daft.udf(return_dtype=daft.DataType.python())
def embed(column, embedding_fn):
    if getattr(embedding_fn, "embed_documents", None) is not None:
        return np.array(embedding_fn.embed_documents(column.to_pylist()))
    return embedding_fn(column.to_pylist())


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
        metadata: Optional[pd.DataFrame] = None,
        embedding_fn=None,
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
            _metadata_dict = metadata.to_dict("list")

        if df is None:
            if isinstance(content, str):
                content = {f"{content}": _metadata_dict.get(content)}
            self.df = daft.from_pydict({**content, **_metadata_dict})
            for k, _ in content.items():
                if embedding_fn is not None:
                    if not k.startswith("embeddings_"):
                        self.df = self.embed(k)
            self.df = self.df.with_column(
                "modal_map_index", indices(col(self.df.column_names[0]))
            )
            if not lazy_start:
                self.df = self.df.collect()

    def _from_plan(self, plan: List[Plan]) -> DaftCollection:
        return DaftCollection(
            df=self.df,
            embedding_fn=self.embedding_fn,
            retriever=self.retriever,
            plan=plan,
        )

    def show(self, num_rows: int):
        return self.df.show(num_rows)

    @property
    def indices(self) -> List[int]:
        return self.df.select("modal_map_index").to_pydict()["modal_map_index"]

    @property
    def column_names(self) -> List[str]:
        return self.df.column_names

    def get_fields(self, column_names: List[str]) -> List[Any]:
        _dict = self.df.to_pydict()
        return [_dict[c] for c in column_names]

    def embed(self, column: str):
        return self.df.with_column(
            f"embeddings_{column}",
            embed(col(column), embedding_fn=self.embedding_fn),
        )

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
            if getattr(self.embedding_fn, "embed_documents"):
                query_embeddings = np.array(self.embedding_fn.embed_documents(query))
            else:
                query_embeddings = self.embedding_fn(query)

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

    def execute_filter(
        self, filter_conditions: Dict[str, Dict[str, str]], *args, **kwargs
    ) -> Collection:
        filtered_df = FilterHelper.filter(self.df, filter_conditions, *args, **kwargs)
        return DaftCollection(
            df=filtered_df,
            embedding_fn=self.embedding_fn,
            retriever=self.retriever,
            plan=self.plan,
        )

    def to_pandas(self) -> pd.DataFrame:
        collection = self.execute()
        return collection.df.to_pandas()

    def save_local(self, directory: str) -> str:
        os.makedirs(directory, exist_ok=True)
        table = self.df.to_arrow()
        pq.write_table(table, os.path.join(directory, "content.parquet"))

    @classmethod
    def from_local_dir(self, local_dir: str, *args, **kwargs) -> DaftCollection:
        df = daft.read_parquet(os.path.join(local_dir, "content.parquet"))
        return DaftCollection(df=df, *args, **kwargs)
