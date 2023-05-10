import abc
from dataclasses import dataclass
from typing import Any, Iterable, Optional

import numpy as np


@dataclass
class QueryOutput:
    embeddings: np.array
    ids: Iterable[Any]
    query_embedding: Optional[np.array] = None


class QueryStrategy(metaclass=abc.ABCMeta):
    @abc.abstractmethod
    def query(
        self,
        query_embeddings: np.array,
        embeddings: np.array,
        ids: np.array,
        *args,
        **kwargs
    ) -> QueryOutput:
        """
        Queries embeddings with query or query embedding vectors and returns nearest embeddings and their corresponding ids


        Args:
            query_embeddings (np.array): query, used to find nearest embeddings in set.
            embeddings (np.array): embeddings set, query is compared to this.
            ids (Iterable[Any]): ids corresponding to the embeddings set.

        Returns:
            QueryOutput: dataclass containing returned embeddings and corresponding ids
        """
