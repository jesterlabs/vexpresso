from __future__ import annotations

from typing import Any, Iterable, Union

import numpy as np


# TODO: support other backends instead of numpy array, like torch
class Embeddings:
    def __init__(self, raw_embeddings: Union[Iterable[Any], str]):
        self.raw_embeddings = raw_embeddings
        if isinstance(self.raw_embeddings, str):
            self.load(self.raw_embeddings)
        elif not isinstance(raw_embeddings, np.ndarray):
            self.raw_embeddings = np.array(raw_embeddings)  # naive way of conversion
        self.raw_embeddings = self.post_process_embeddings(self.raw_embeddings)

    @property
    def shape(self):
        return self.raw_embeddings.shape

    @classmethod
    def from_raw(cls, raw_embeddings: Iterable[Any], *args, **kwargs):
        return cls(raw_embeddings, *args, **kwargs)

    def post_process_embeddings(self, raw_embeddings: np.ndarray) -> np.ndarray:
        raw_embeddings = np.squeeze(raw_embeddings)
        if len(raw_embeddings.shape) == 1:
            return np.expand_dims(raw_embeddings, axis=0)
        return raw_embeddings

    def add(self, embeddings: Union[Any, Embeddings]) -> Embeddings:
        other = embeddings
        if not isinstance(embeddings, Embeddings):
            other = Embeddings(
                raw_embeddings=embeddings,
            )
        return Embeddings(
            np.concatenate((self.raw_embeddings, other.raw_embeddings), axis=0)
        )

    def __len__(self) -> int:
        return self.raw_embeddings.shape[0]

    def index(self, indices: Iterable[int]) -> Embeddings:
        return Embeddings(self.raw_embeddings[indices])

    def save(self, path: str) -> str:
        with open(path, "wb") as f:
            np.save(f, self.raw_embeddings)
        return path

    def load(self, path: str):
        with open(path, "rb") as f:
            self.raw_embeddings = np.load(f)
