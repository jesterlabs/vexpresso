from __future__ import annotations

from typing import Any, Iterable, List, Optional, Union

import numpy as np

from vexpresso.embedding_function import EmbeddingFunction
from vexpresso.query import NumpyQueryStrategy, QueryOutput, QueryStrategy


# TODO: support other backends instead of numpy array, like torch
class Embeddings:
    def __init__(
        self,
        embedding_vectors: Optional[np.ndarray] = None,
        embedding_fn: EmbeddingFunction = None,
        query_strategy: QueryStrategy = NumpyQueryStrategy(),
        content: Optional[Iterable[Any]] = None,
    ):
        self.embedding_vectors = embedding_vectors
        self.embedding_fn = embedding_fn
        self.query_strategy = query_strategy

        if content is None and self.embedding_vectors is None:
            raise ValueError("Either content or embeddings must be specified!")

        if content is not None:
            if self.embedding_vectors is None:
                self.embedding_vectors = self.embedding_fn(content)

        self.embedding_vectors = self.post_process_embeddings(self.embedding_vectors)

    def post_process_embeddings(self, embedding_vectors: np.ndarray) -> np.ndarray:
        if len(embedding_vectors.shape) == 1:
            return np.expand_dims(embedding_vectors, axis=0)
        return embedding_vectors

    def __len__(self) -> int:
        return self.embedding_vectors.shape[0]

    def __getitem__(self, key) -> Embeddings:
        if isinstance(key, int):
            return self._getitem(key)
        elif isinstance(key, slice):
            return self._getslice(key)
        elif isinstance(key, Iterable):
            return self._getiterable(key)
        else:
            raise TypeError("Index must be int, not {}".format(type(key).__name__))

    def _getitem(self, idx: int) -> Embeddings:
        return Embeddings(
            embedding_vectors=self.embedding_vectors[idx : idx + 1],
            embedding_fn=self.embedding_fn,
            query_strategy=self.query_strategy,
        )

    def _getiterable(self, indices: Iterable[int]) -> Embeddings:
        return Embeddings(
            embedding_vectors=self.embedding_vectors[indices],
            embedding_fn=self.embedding_fn,
            query_strategy=self.query_strategy,
        )

    def _getslice(self, index_slice: slice) -> Embeddings:
        return Embeddings(
            embedding_vectors=self.embedding_vectors[
                index_slice.start : index_slice.stop
            ],
            embedding_fn=self.embedding_fn,
            query_strategy=self.query_strategy,
        )

    def query(
        self,
        query: Any = None,
        query_embedding: Optional[np.ndarray] = None,
        *args,
        **kwargs,
    ) -> Union[List[QueryOutput], QueryOutput]:
        """
        Queries embeddings with query or query embedding vectors and returns nearest embeddings and their content

        Args:
            query (Any): query to embed and compare to embeddings in set. Defaults to None.
            query_embedding (np.ndarray, optional): vector embedding of query to compare to. Defaults to None.

        Returns:
            EmbeddingQueryOutput: Dataclass that contains embedding information, embedding ids, and queries
        """
        if query_embedding is None:
            # TODO: maybe explicitly call batch function here?
            query_embedding = self.embedding_fn(query)
        query_output = self.query_strategy.query(
            query_embedding, self.embedding_vectors, *args, **kwargs
        )
        return query_output

    def __call__(self, *args, **kwargs) -> np.ndarray:
        return self.embedding_fn(*args, **kwargs)
