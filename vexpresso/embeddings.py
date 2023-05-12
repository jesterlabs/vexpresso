from __future__ import annotations

from typing import Any, Iterable, List, Optional, Union

import numpy as np


collection = Collection(embeddings, metadata, emb)


collection = vexpresso.create_collection(content, embedding_fn, metadata, backend="numpy")

class Collection:

    ACCEPTED_TYPES = ...



    def embed(self, ):

    def query(self, )


# TODO: support other backends instead of numpy array, like torch
class Embeddings:
    def __init__(
        self,
        embedding_vectors: Optional[np.ndarray] = None,
    ):
        self.embedding_vectors = embedding_vectors
        self.embedding_vectors = self.post_process_embeddings(self.embedding_vectors)

    def post_process_embeddings(self, embedding_vectors: np.ndarray) -> np.ndarray:
        if len(embedding_vectors.shape) == 1:
            return np.expand_dims(embedding_vectors, axis=0)
        return embedding_vectors

    def __add__(self, other: Embeddings) -> Embeddings:
        embedding_vectors = np.concatenate(
            (self.embedding_vectors, other.embedding_vectors), axis=0
        )
        return Embeddings(embedding_vectors)

    def append(self, other: Embeddings) -> Embeddings:
        self.embedding_vectors = np.concatenate(
            (self.embedding_vectors, other.embedding_vectors), axis=0
        )
        return self

    def add(
        self,
        embedding_vectors: Optional[np.ndarray] = None,
    ) -> Embeddings:
        other = Embeddings(
            embedding_vectors=embedding_vectors,
        )
        self.append(other)
        return self

    def __len__(self) -> int:
        return self.embedding_vectors.shape[0]

    def index(self, key) -> Embeddings:
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
        )

    def _getiterable(self, indices: Iterable[int]) -> Embeddings:
        return Embeddings(
            embedding_vectors=self.embedding_vectors[indices],
        )

    def _getslice(self, index_slice: slice) -> Embeddings:
        return Embeddings(
            embedding_vectors=self.embedding_vectors[
                index_slice.start : index_slice.stop
            ],
        )
