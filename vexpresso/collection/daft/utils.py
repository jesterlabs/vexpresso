from __future__ import annotations

import inspect
from functools import wraps
from typing import Any, Dict, List

import daft

from vexpresso.collection.collection import Collection
from vexpresso.utils import Transformation


def lazy(default: bool = True):
    def dec(func):
        @wraps(func)
        def wrapper(*args, lazy=default, **kwargs) -> Collection:
            collection = func(*args, **kwargs)
            if not lazy:
                collection = collection.execute()
            return collection

        return wrapper

    return dec


def convert_args(*args):
    return [
        arg.to_pylist() if isinstance(arg, daft.series.Series) else arg for arg in args
    ]


def convert_kwargs(**kwargs):
    return {
        k: kwargs[k].to_pylist()
        if isinstance(kwargs[k], daft.series.Series)
        else kwargs[k]
        for k in kwargs
    }


# TODO: CHANGE TO ENUM
DATATYPES = {"python": daft.DataType.python}


def transformation(
    original_function: Transformation = None, *, datatype: str = "python"
):
    def _decorate(function: Transformation):
        @wraps(function)
        def wrapped(*args, **kwargs):
            args = convert_args(*args)
            kwargs = convert_kwargs(**kwargs)
            return function(*args, **kwargs)

        wrapped.__signature__ = inspect.signature(function)

        daft_datatype = DATATYPES.get(datatype, DATATYPES["python"])
        _udf = daft.udf(return_dtype=daft_datatype())(wrapped)
        _udf.__vexpresso_transform = True
        return _udf

    if original_function:
        return _decorate(original_function)

    return _decorate


class Scope:
    def __init__(self, columns: List[str], collection: Collection):
        self.columns = columns
        self.collection = collection

    def col(self, *args) -> Scope:
        return Scope(columns=args, collection=self.collection)

    @lazy(default=True)
    def select(
        self,
    ) -> Scope:
        return Scope(
            columns=self.columns,
            collection=self.collection.select(
                *self.columns,
            ),
        )

    @lazy(default=True)
    def query(
        self,
        query: Any = None,
        query_embeddings: Any = None,
        filter_conditions: Dict[str, str] = None,
        *args,
        **kwargs,
    ) -> Scope:
        if len(self.columns) > 1:
            raise ValueError("Query method can only be used with one column!")
        if query is None:
            query = {}
        else:
            query = {column: query for column in self.columns}

        if query_embeddings is None:
            query_embeddings = {}
        else:
            query_embeddings = {column: query_embeddings for column in self.columns}

        if filter_conditions is not None:
            filter_conditions = {column: filter_conditions for column in self.columns}

        return Scope(
            self.columns,
            self.collection.query(
                query, query_embeddings, filter_conditions, *args, **kwargs
            ),
        )

    def execute(self) -> Scope:
        return Scope(self.columns, self.collection.execute())

    @lazy(default=True)
    def filter(self, filter_conditions: Dict[str, str], *args, **kwargs) -> Scope:
        filter_conditions = {column: filter_conditions for column in self.columns}
        return Scope(
            self.columns, self.collection.filter(filter_conditions, *args, **kwargs)
        )

    @lazy(default=True)
    def apply(
        self,
        transform_fn: Transformation,
        to: str,
        *args,
        **kwargs,
    ) -> Scope:
        collection = self.collection
        for column in self.columns:
            collection = collection.apply(column, transform_fn, to, *args, **kwargs)
        return Scope(self.columns, collection)

    @lazy(default=True)
    def embed(
        self,
        embedding_fn: Transformation = None,
        *args,
        **kwargs,
    ) -> Scope:
        # reset embedding_fn
        collection = self.collection
        for column in self.columns:
            collection = collection.embed(column, embedding_fn, *args, **kwargs)
        return Scope(self.columns, collection)
