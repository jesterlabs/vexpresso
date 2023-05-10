import abc


class BaseMetadata(metaclass=abc.ABCMeta):
    @abc.abstractmethod
    def query(self, *args, **kwargs):
        """
        Abstract method for querying metadata
        """
