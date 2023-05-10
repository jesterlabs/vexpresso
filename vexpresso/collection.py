from __future__ import annotations

from typing import Any, Callable, Iterable, List, Optional, Union

import numpy as np

from vexpresso.embeddings import Embeddings
from vexpresso.query import QueryOutput, QueryStrategy
from vexpresso.strategy import NumpyStrategy


class Collection:
    def __init__(
        self,
        content: Iterable[Any] = None,
        embeddings: Union[np.ndarray, Embeddings] = None,
        ids: Optional[Iterable[Any]] = None,
        embedding_fn: Callable[[Any], np.ndarray] = None,
        lookup_strategy: QueryStrategy = NumpyStrategy(),
    ):
        self.ids = ids
        self.embeddings = embeddings

        # if embeddings is not provided or if a numpy array is provided
        if self.embeddings is None or isinstance(self.embeddings, np.ndarray):
            self.embeddings = Embeddings(
                content, self.embeddings, embedding_fn, lookup_strategy
            )

        if content is not None:
            # set content
            self.embeddings.content = content

        if self.ids is None:
            self.ids = list(range(len(self.embeddings)))

        self.assert_data_types()

    def assert_data_types(self):
        # TODO: probably improve this or remove this logic entirely
        if not isinstance(self.embeddings, Embeddings):
            raise ValueError(
                "embeddings must either be provided as a numpy array or as an Embeddings object"
            )

    @property
    def content(self) -> List[Any]:
        return self.embeddings.content

    @property
    def embedding_fn(self):
        return self.embeddings.embedding_fn

    @property
    def lookup_strategy(self):
        return self.embeddings.lookup_strategy

    @property
    def embedding_vectors(self):
        return self.embeddings.embeddings

    def query(
        self,
        query: List[Any] = None,
        query_embedding: Optional[np.ndarray] = None,
        return_collection: bool = True,
        *args,
        **kwargs,
    ) -> Union[Collection, QueryOutput]:
        query_output = self.embeddings.query(query, query_embedding, *args, **kwargs)
        if return_collection:
            embeddings = Embeddings(
                query_output.embeddings,
                self.embedding_fn,
                self.lookup_strategy,
            )
            indices = query_output.indices
            content = [content[idx] for idx in indices]
            ids = [ids[idx] for idx in indices]
            return Collection(
                content,
                embeddings,
                ids,
            )
        return query_output
