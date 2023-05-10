from __future__ import annotations

from typing import Any, Callable, Iterable, Optional

import numpy as np

from vexpresso.query.query import NumpyStrategy, QueryOutput, QueryStrategy


class Embeddings:
    def __init__(
        self,
        embeddings: np.array,
        ids: Iterable[Any],
        embedding_fn: Callable[[Any], np.array],
        lookup_strategy: QueryStrategy = NumpyStrategy(),
    ):
        self.embeddings = embeddings
        self.ids = ids
        self.embedding_fn = embedding_fn
        self.lookup_strategy = lookup_strategy

    def query(
        self,
        query: Iterable[Any] = None,
        query_embedding: Optional[Iterable[Any]] = None,
        batch: bool = False,
        *args,
        **kwargs,
    ) -> QueryOutput:
        """
        Queries embeddings with query or query embedding vectors and returns nearest embeddings and their content

        Args:
            query (Iterable[Any]): query to embed and compare to embeddings in set. Defaults to None.
            query_embedding (Iterable[Any], optional): vector embedding of query to compare to. Defaults to None.
            k (int, optional): how many nearest embeddings to return. Defaults to 4.

        Returns:
            EmbeddingQueryOutput: Dataclass that contains embedding information, embedding ids, and queries
        """
        if query_embedding is None:
            if batch:
                query_embedding = np.stack(self.embedding_fn(q) for q in query)
            else:
                query_embedding = self.embedding_fn(query)
        return self.lookup_strategy.query(
            query_embedding, self.embeddings, self.ids, *args, **kwargs
        )
