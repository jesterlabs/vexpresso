import abc
from typing import Any, List

from vexpresso.utils import Transformation, transformation


def get_embedding_fn(embedding_fn: Transformation) -> Transformation:
    # langchain check
    if getattr(embedding_fn, "embed_documents", None) is not None:
        return transformation(embedding_fn, function="embed_documents")
    return transformation(embedding_fn)


class EmbeddingFunction(metaclass=abc.ABCMeta):
    @abc.abstractmethod
    def __call__(self, column: List[Any], *args, **kwargs):
        """
        This is the main function of `embedding function` to be applied on a column
        """
