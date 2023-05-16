import abc
from dataclasses import dataclass
from typing import Any, Iterable, List, Optional, Union

import numpy as np


@dataclass
class RetrievalOutput:
    embeddings: Any
    indices: Union[np.ndarray, Iterable[int]]
    query_embeddings: Optional[Any] = None


class RetrievalStrategy(metaclass=abc.ABCMeta):
    SUPPORTED_TYPES = [np.dtype]

    @abc.abstractmethod
    def retrieve(
        self, query_embedding: np.ndarray, embeddings: np.ndarray, *args, **kwargs
    ) -> Union[List[RetrievalOutput], RetrievalOutput]:
        """
        Queries embeddings with query embedding vector and returns nearest embeddings and their corresponding ids


        Args:
            query_embedding (np.ndarray): query, used to find nearest embeddings in set.
            embeddings (np.ndarray): embeddings set, query is compared to this.

        Returns:
            Union[List[QueryOutput], QueryOutput]: dataclasses containing returned embeddings and corresponding indices.
            When this has more than one entry, that means that the call was batched
        """
