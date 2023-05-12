import abc
from dataclasses import dataclass
from typing import Any, Iterable, List, Optional, Union

import numpy as np


@dataclass
class QueryOutput:
    embeddings: np.ndarray
    indices: Union[np.ndarray, Iterable[int]]
    query_embeddings: Optional[np.ndarray] = None

    # content is populated by collections class
    content: Optional[List[Any]] = None


class QueryStrategy(metaclass=abc.ABCMeta):
    @abc.abstractmethod
    def query(
        self, query_embedding: np.ndarray, embeddings: np.ndarray, *args, **kwargs
    ) -> Union[List[QueryOutput], QueryOutput]:
        """
        Queries embeddings with query embedding vector and returns nearest embeddings and their corresponding ids


        Args:
            query_embedding (np.ndarray): query, used to find nearest embeddings in set.
            embeddings (np.ndarray): embeddings set, query is compared to this.

        Returns:
            Union[List[QueryOutput], QueryOutput]: dataclasses containing returned embeddings and corresponding indices.
            When this has more than one entry, that means that the call was batched
        """
