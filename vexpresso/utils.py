from __future__ import annotations

import inspect
import os
from collections.abc import Iterable
from dataclasses import dataclass, field
from functools import reduce, wraps
from typing import Any, Callable, Dict, List, Optional, Tuple

import daft
import numpy as np

ResourceRequest = daft.resource_request.ResourceRequest
DataType = daft.datatype.DataType


def get_batch_size(embeddings: Iterable[Any]) -> int:
    if isinstance(embeddings, np.ndarray):
        if len(embeddings.shape) == 1:
            return 1
    return len(embeddings)


# LANGCHAIN
@dataclass
class Document:
    """Interface for interacting with a document."""

    page_content: str
    metadata: dict = field(default_factory=dict)


def batchify_args(args, batch_size):
    if isinstance(args, Iterable) and not isinstance(args, str):
        if len(args) != batch_size:
            raise ValueError("ARG needs to be size batch size")
    else:
        args = [args for _ in range(batch_size)]
    return args


def lazy(default: bool = True):
    def dec(func):
        @wraps(func)
        def wrapper(*args, lazy=default, **kwargs):
            collection = func(*args, **kwargs)
            if not lazy:
                if isinstance(collection, list):
                    collection = [c.execute() for c in collection]
                else:
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


def transform_wrapper(
    original_transform: Transformation = None,
    datatype: DataType = DataType.python(),
    init_kwargs: Dict[str, Any] = {},
    function: str = "__call__",
):
    if inspect.isfunction(original_transform):

        def _decorate(function: Transformation):
            @wraps(function)
            def wrapped(*args, **kwargs):
                args = convert_args(*args)
                kwargs = convert_kwargs(**kwargs)
                return function(*args, **kwargs)

            wrapped.__signature__ = inspect.signature(function)

            _udf = daft.udf(return_dtype=datatype)(wrapped)
            _udf.__vexpresso_transform = True
            return _udf

        return _decorate(original_transform)
    else:
        if isinstance(original_transform, type):
            sig = inspect.signature(getattr(original_transform, function))
        else:
            sig = inspect.signature(getattr(original_transform.__class__, function))

        class _Transformation:
            def __init__(self):
                if isinstance(original_transform, type):
                    # hasn't been initialized yet
                    self._transform = original_transform(**init_kwargs)
                else:
                    self._transform = original_transform

            def __call__(self, *args, **kwargs):
                args = convert_args(*args)
                kwargs = convert_kwargs(**kwargs)
                return getattr(self._transform, function)(*args, **kwargs)

        _Transformation.__call__.__signature__ = sig
        _udf = daft.udf(return_dtype=datatype)(_Transformation)
        _udf.__vexpresso_transform = True
        return _udf


def transformation(
    original_function: Transformation = None,
    datatype: DataType = DataType.python(),
    init_kwargs={},
    function: str = "__call__",
):
    if getattr(original_function, "__vexpresso_transform", None) is None:
        wrapper = transform_wrapper(
            original_function,
            datatype=datatype,
            init_kwargs=init_kwargs,
            function=function,
        )
        return wrapper
    return original_function


def get_field_name_and_key(field: str, column_names: List[str] = []) -> Tuple[str, str]:
    if field in column_names:
        return field, None

    field_name = field.split(".")[0]

    keys = None
    if "." in field:
        keys = field.split(".", 1)[-1]
    return field_name, keys


def deep_get(dictionary, keys=None, default=None):
    if keys is None:
        return dictionary
    if isinstance(dictionary, dict):
        return reduce(
            lambda d, key: d.get(key, default) if isinstance(d, dict) else default,
            keys.split("."),
            dictionary,
        )
    return dictionary


class HFHubHelper:
    def __init__(self):
        self._hf_hub = None
        try:
            import huggingface_hub  # noqa

            self._hf_hub = huggingface_hub
        except ImportError:
            raise ImportError(
                "Could not import huggingface_hub python package."
                "Please install it with `pip install huggingface_hub`."
            )
        self.api = self._hf_hub.HfApi()

    def create_repo(
        self, repo_id: str, token: Optional[str] = None, *args, **kwargs
    ) -> str:
        if token is None:
            token = os.environ.get("HUGGINGFACEHUB_API_TOKEN")
        return self.api.create_repo(
            repo_id=repo_id,
            token=token,
            exist_ok=True,
            repo_type="dataset",
            *args,
            **kwargs,
        )

    def upload(
        self,
        repo_id: str,
        folder_path: str,
        token: Optional[str] = None,
        private: bool = True,
        *args,
        **kwargs,
    ) -> str:
        if token is None:
            token = os.environ.get("HUGGINGFACEHUB_API_TOKEN")
        self.create_repo(repo_id, token, private=private)
        return self.api.upload_folder(
            repo_id=repo_id,
            folder_path=folder_path,
            token=token,
            repo_type="dataset",
            *args,
            **kwargs,
        )

    def download(
        self,
        repo_id: str,
        token: Optional[str] = None,
        local_dir: Optional[str] = None,
        *args,
        **kwargs,
    ) -> str:
        if token is None:
            token = os.environ.get("HUGGINGFACEHUB_API_TOKEN")
        return self._hf_hub.snapshot_download(
            repo_id=repo_id,
            token=token,
            local_dir=local_dir,
            repo_type="dataset",
            *args,
            **kwargs,
        )


Transformation = Callable[[List[Any], Any], List[Any]]
