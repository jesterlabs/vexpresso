from __future__ import annotations

from typing import Any, Callable, Iterable, List, Optional, Union

import numpy as np

from vexpresso.embedding_function import EmbeddingFunction
from vexpresso.embeddings import Embeddings
from vexpresso.query import NumpyQueryStrategy, QueryOutput, QueryStrategy


class Collection:
    def __init__(
        self,
        content: Iterable[Any] = None,
        embeddings: Union[np.ndarray, Embeddings] = None,
        ids: Optional[Iterable[Any]] = None,
        embedding_fn: Union[
            EmbeddingFunction, Callable[[Iterable[Any]], np.ndarray]
        ] = None,
        query_strategy: QueryStrategy = NumpyQueryStrategy(),
    ):
        self.content = content
        self.embeddings = embeddings
        self.ids = ids

        # if embeddings is not provided or if a numpy array is provided
        if self.embeddings is None or isinstance(self.embeddings, np.ndarray):
            self.embeddings = Embeddings(
                self.embeddings, embedding_fn, query_strategy, self.content
            )

        if self.content is None:
            self.content = [None for _ in range(len(self.embeddings))]

        if self.ids is None:
            self.ids = list(range(len(self.embeddings)))

        self.assert_data_types()

    def assert_data_types(self):
        # TODO: probably improve this or remove this logic entirely
        if not isinstance(self.embeddings, Embeddings):
            raise ValueError(
                "embeddings must either be provided as a numpy array or as an Embeddings object"
            )

    def __len__(self) -> int:
        return len(self.embeddings)

    @property
    def embedding_fn(self):
        return self.embeddings.embedding_fn

    @property
    def query_strategy(self):
        return self.embeddings.query_strategy

    @property
    def embedding_vectors(self):
        return self.embeddings.embedding_vectors

    def __getitem__(self, key) -> Collection:
        if isinstance(key, int):
            return self._getitem(key)
        elif isinstance(key, slice):
            return self._getslice(key)
        elif isinstance(key, Iterable):
            return self._getiterable(key)
        else:
            raise TypeError("Index must be int, not {}".format(type(key).__name__))

    def _getitem(self, idx: int) -> Collection:
        return Collection(
            self.content[idx : idx + 1],
            self.embeddings[idx],
            self.ids[idx : idx + 1],
            self.embedding_fn,
            self.query_strategy,
        )

    def _getiterable(self, indices: Iterable[int]) -> Collection:
        return Collection(
            [self.content[idx] for idx in indices],
            self.embeddings[indices],
            [self.ids[idx] for idx in indices],
            self.embedding_fn,
            self.query_strategy,
        )

    def _getslice(self, index_slice: slice) -> Collection:
        return Collection(
            self.content[index_slice.start : index_slice.stop],
            self.embeddings[index_slice],
            self.ids[index_slice.start : index_slice.stop],
            self.embedding_fn,
            self.query_strategy,
        )

    def query(
        self,
        query: List[Any] = None,
        query_embedding: Optional[np.ndarray] = None,
        return_collection: bool = True,
        *args,
        **kwargs,
    ) -> Union[Collection, QueryOutput, List[Collection], List[QueryOutput]]:
        query_output = self.embeddings.query(query, query_embedding, *args, **kwargs)

        if not return_collection:
            return query_output

        if isinstance(query_output, QueryOutput):
            # not batched
            return self[query_output.indices]

        if len(query_output) == 0:
            return self[query_output[0].indices]

        # batched calls
        collection_list = []
        for q in query_output:
            collection_list.append(self[q.indices])
        return collection_list
