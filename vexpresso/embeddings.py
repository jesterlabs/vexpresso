from __future__ import annotations

from typing import Any, Callable, List, Optional

import numpy as np

from vexpresso.query.query import NumpyStrategy, QueryOutput, QueryStrategy


# TODO: support other backends instead of numpy array, like torch
class Embeddings:
    def __init__(
        self,
        content: List[Any] = None,
        embeddings: Optional[np.ndarray] = None,
        embedding_fn: Optional[Callable[[Any], np.ndarray]] = None,
        lookup_strategy: QueryStrategy = NumpyStrategy(),
    ):
        self.content = content
        self.embeddings = embeddings
        self.embedding_fn = embedding_fn
        self.lookup_strategy = lookup_strategy

        if self.content is None and self.embeddings is None:
            raise ValueError("Either content or embeddings must be specified!")

        if self.content is not None:
            if self.embeddings is None:
                self.embeddings = self.embedding_fn(self.content)

        if self.content is None:
            self.content = [None for _ in range(len(self))]

    def __len__(self):
        return self.embeddings.shape[0]

    def query(
        self,
        query: List[Any] = None,
        query_embedding: Optional[np.ndarray] = None,
        *args,
        **kwargs,
    ) -> QueryOutput:
        """
        Queries embeddings with query or query embedding vectors and returns nearest embeddings and their content

        Args:
            query (List[Any]): query to embed and compare to embeddings in set. Defaults to None.
            query_embedding (Iterable[Any], optional): vector embedding of query to compare to. Defaults to None.
            k (int, optional): how many nearest embeddings to return. Defaults to 4.

        Returns:
            EmbeddingQueryOutput: Dataclass that contains embedding information, embedding ids, and queries
        """
        if query_embedding is None:
            query_embedding = self.embed(query)
        query_output = self.lookup_strategy.query(
            query_embedding, self.embeddings, *args, **kwargs
        )
        # filter content
        # TODO: make this work for batched scenarios
        query_output.content = [self.content[idx] for idx in query_output.indices]
        return query_output

    def add(
        self,
        content: Optional[List[Any]] = None,
        embeddings: Optional[np.ndarray] = None,
    ):
        if content is None and embeddings is None:
            raise ValueError("Either content or embeddings must be specified!")
        if content is not None:
            if embeddings is None:
                embeddings = self.embed(content)
        else:
            content = [None for _ in range(embeddings.shape[0])]
        # concatenate list
        self.content = self.content + content
        # stack embedding arrays.
        # TODO: This should be framework agnostic
        self.embeddings = np.concatenate((self.embeddings, embeddings), axis=0)

    def embed(self, content: List[Any]) -> np.ndarray:
        return np.stack([self.embedding_fn(c) for c in content])

    def __call__(self, *args, **kwargs) -> np.ndarray:
        return self.embed(*args, **kwargs)
