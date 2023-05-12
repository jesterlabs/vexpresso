import abc
from typing import Any


class EmbeddingFunction(metaclass=abc.ABCMeta):
    @abc.abstractmethod
    def embed(self, inputs: Any, *args, **kwargs) -> Any:
        """
        embeds inputs
        """

    def __call__(self, inputs: Any, *args, **kwargs) -> Any:
        return self.embed(inputs, *args, **kwargs)
