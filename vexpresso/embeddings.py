from __future__ import annotations

import os
from typing import Any, Iterable, Optional, Union

import numpy as np


# TODO: support other backends instead of numpy array, like torch
class Embeddings:
    def __init__(
        self, raw_embeddings: Iterable[Any] = None, saved_path: Optional[str] = None
    ):
        if saved_path is not None:
            self.load(saved_path)
        else:
            self.raw_embeddings = raw_embeddings
            if not isinstance(raw_embeddings, np.ndarray):
                self.raw_embeddings = np.array(
                    raw_embeddings
                )  # naive way of conversion
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

    def append(self, other: Embeddings) -> Embeddings:
        self.raw_embeddings = np.concatenate(
            (self.raw_embeddings, other.raw_embeddings), axis=0
        )
        return self

    def add(self, embeddings: Union[Any, Embeddings]) -> Embeddings:
        other = embeddings
        if not isinstance(embeddings, Embeddings):
            other = Embeddings(
                raw_embeddings=embeddings,
            )
        self.append(other)
        return self

    def __len__(self) -> int:
        return self.raw_embeddings.shape[0]

    def index(self, indices: Iterable[int]) -> Embeddings:
        return Embeddings(self.raw_embeddings[indices])

    def save(self, path: str, filename: Optional[str] = None) -> str:
        if filename is None:
            filename = "embeddings.npy"
        path = os.path.join(path, filename)
        with open(path, "wb") as f:
            np.save(f, self.raw_embeddings)
        return path

    def load(self, path: str, filename: Optional[str] = None):
        if filename is None:
            filename = "embeddings.npy"
        path = os.path.join(path, filename)
        with open(path, "rb") as f:
            self.raw_embeddings = np.load(f)
