import abc
from typing import Any, Dict, List

from vexpresso.utils import DataType, Transformation, transformation


def get_embedding_fn(
    embedding_fn: Transformation,
    datatype: DataType = DataType.python(),
    init_kwargs: Dict[str, Any] = {},
) -> Transformation:
    # langchain check
    if getattr(embedding_fn, "embed_documents", None) is not None:
        return transformation(
            embedding_fn,
            datatype=datatype,
            function="embed_documents",
            init_kwargs=init_kwargs,
        )
    return transformation(embedding_fn, datatype=datatype, init_kwargs=init_kwargs)


class EmbeddingFunction(metaclass=abc.ABCMeta):
    @abc.abstractmethod
    def __call__(self, column: List[Any], *args, **kwargs):
        """
        This is the main function of `embedding function` to be applied on a column
        """
