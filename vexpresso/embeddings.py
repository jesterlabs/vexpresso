from __future__ import annotations

from typing import Iterable, Optional

import numpy as np


# TODO: support other backends instead of numpy array, like torch
class Embeddings:
    def __init__(
        self,
        raw_embeddings: Optional[np.ndarray] = None,
    ):
        self.raw_embeddings = raw_embeddings
        self.raw_embeddings = self.post_process_embeddings(self.raw_embeddings)

    def post_process_embeddings(self, raw_embeddings: np.ndarray) -> np.ndarray:
        if len(raw_embeddings.shape) == 1:
            return np.expand_dims(raw_embeddings, axis=0)
        return raw_embeddings

    def append(self, other: Embeddings) -> Embeddings:
        self.raw_embeddings = np.concatenate(
            (self.raw_embeddings, other.raw_embeddings), axis=0
        )
        return self

    def add(
        self,
        raw_embeddings: Optional[np.ndarray] = None,
    ) -> Embeddings:
        other = Embeddings(
            raw_embeddings=raw_embeddings,
        )
        self.append(other)
        return self

    def __len__(self) -> int:
        return self.raw_embeddings.shape[0]

    def index(self, indices: Iterable[int]) -> Embeddings:
        return Embeddings(self.raw_embeddings[indices])
