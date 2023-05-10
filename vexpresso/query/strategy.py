import abc
from dataclasses import dataclass
from typing import Any, List, Optional

import numpy as np


@dataclass
class QueryOutput:
    embeddings: np.array
    indices: List[int]
    query_embedding: Optional[np.array] = None

@dataclass
class BatchedQueryOutput:
    embeddings: np.array
    indices: List[List[int]]
    query_embedding: Optional[np.array] = None


class QueryStrategy(metaclass=abc.ABCMeta):
    @abc.abstractmethod
    def query(
        self,
        query_embeddings: np.array,
        embeddings: np.array,
        *args,
        **kwargs
    ) -> QueryOutput:
        """
        Queries embeddings with query or query embedding vectors and returns nearest embeddings and their corresponding ids


        Args:
            query_embeddings (np.array): query, used to find nearest embeddings in set.
            embeddings (np.array): embeddings set, query is compared to this.

        Returns:
            QueryOutput: dataclass containing returned embeddings and corresponding indices
        """
