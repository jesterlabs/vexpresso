from __future__ import annotations

from typing import Any, Callable, Iterable, Optional, Union

import numpy as np

from vexpresso.embeddings import Embeddings
from vexpresso.query import QueryOutput, QueryStrategy
from vexpresso.strategy import NumpyStrategy


class Collection:
    def __init__(
        self,
        embeddings: Union[np.array, Embeddings],
        ids: Optional[Iterable[Any]] = None,
        embedding_fn: Callable[[Any], np.array] = None,
        lookup_strategy: QueryStrategy = NumpyStrategy(),
    ):
        self.embeddings = embeddings
        if isinstance(embeddings, np.ndarray):
            if ids is None:
                ids = list(range(embeddings.shape[0]))
            self.embeddings = Embeddings(embeddings, ids, embedding_fn, lookup_strategy)

    @classmethod
    def from_embeddings(cls, embeddings: Embeddings, *args, **kwargs) -> Collection:
        return Collection(
            embeddings.embeddings,
            embeddings.ids,
            embeddings.embedding_fn,
            embeddings.lookup_strategy,
            *args,
            **kwargs,
        )

    def query(
        self,
        query: Iterable[Any] = None,
        query_embedding: Optional[Iterable[Any]] = None,
        batch: bool = False,
        return_collection: bool = True,
        *args,
        **kwargs,
    ) -> Union[Collection, QueryOutput]:
        query_output = self.embeddings.query(
            query, query_embedding, batch, *args, **kwargs
        )
        if return_collection:
            embeddings = Embeddings(
                query_output.embeddings,
                query_output.ids,
                self.embeddings.embedding_fn,
                self.embeddings.lookup_strategy,
            )
            return Collection.from_embeddings(embeddings)
        return query_output
