import abc
from typing import Any, Callable, List


class Transformation(metaclass=abc.ABCMeta):
    @abc.abstractmethod
    def transform(self, content: List[Any]) -> List[Any]:
        """
        Transform method, returns a transformed list of the same shape

        Args:
            content (List[Any]): list to be transformed

        Returns:
            List[Any]: transformed list
        """

    def __call__(self, *args, **kwargs) -> List[Any]:
        return self.transform(*args, **kwargs)


class CallableTransformation(Transformation):
    def __init__(self, transform_func: Callable[[List[Any]], List[Any]]):
        self.transform_func = transform_func

    def transform(self, content: List[Any]) -> List[Any]:
        return self.transform_func(content)
