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
