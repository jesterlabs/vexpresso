import abc
from dataclasses import dataclass
from typing import List, Optional, Union

import numpy as np


@dataclass
class QueryOutput:
    embeddings: np.ndarray
    indices: List[int]
    query_embeddings: Optional[np.ndarray] = None


@dataclass
class BatchedQueryOutput:
    embeddings: np.ndarray
    indices: List[List[int]]
    query_embeddings: Optional[Union[List[np.ndarray], np.ndarray]] = None


class QueryStrategy(metaclass=abc.ABCMeta):
    @abc.abstractmethod
    def query(
        self, query_embedding: np.ndarray, embeddings: np.ndarray, *args, **kwargs
    ) -> QueryOutput:
        """
        Queries embeddings with query embedding vector and returns nearest embeddings and their corresponding ids


        Args:
            query_embedding (np.ndarray): query, used to find nearest embeddings in set.
            embeddings (np.ndarray): embeddings set, query is compared to this.

        Returns:
            QueryOutput: dataclass containing returned embeddings and corresponding indices
        """

    def batched_query(
        self, query_embeddings: np.ndarray, embeddings: np.ndarray, *args, **kwargs
    ) -> BatchedQueryOutput:
        """
        Queries embeddings with query embedding vectors and returns nearest embeddings and their corresponding ids.
        This is a batched version, supporting multiple query embeddings.
        By default we just loop over the query_embeddings, but this can be overridden for more efficient implementations.


        Args:
            query_embeddings (np.ndarray): query, used to find nearest embeddings in set.
            embeddings (np.ndarray): embeddings set, query is compared to this.

        Returns:
            BatchedQueryOutput: dataclass containing returned embeddings and corresponding indices

        """
        outputs = []
        for q in query_embeddings:
            query_output = self.query(q, embeddings, *args, **kwargs)
            outputs.append(query_output)

        batched_embeddings = np.stack(
            [query_output.embeddings for query_output in outputs]
        )
        batched_indices = [query_output.indices for query_output in outputs]
        batched_query_embeddings = [
            query_output.query_embeddings for query_output in outputs
        ]
        return BatchedQueryOutput(
            batched_embeddings, batched_indices, batched_query_embeddings
        )
