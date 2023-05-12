from __future__ import annotations

import uuid
from typing import Any, Callable, Dict, Iterable, List, Optional, Union

import numpy as np
import pandas as pd

from vexpresso.embedding_function import EmbeddingFunction
from vexpresso.embeddings import Embeddings
from vexpresso.metadata import Metadata
from vexpresso.query import NumpyQueryStrategy, QueryOutput, QueryStrategy


class Collection:
    def __init__(
        self,
        content: List[Any] = None,
        embeddings: Union[np.ndarray, Embeddings] = None,
        ids: Optional[List[str]] = None,
        metadata: Optional[Union[pd.DataFrame, Metadata, Dict[str, Any]]] = None,
        embedding_fn: Union[
            EmbeddingFunction, Callable[[Iterable[Any]], np.ndarray]
        ] = None,
        query_strategy: QueryStrategy = NumpyQueryStrategy(),
    ):
        self.content = content
        self.embeddings = embeddings
        self.metadata = metadata
        self.query_strategy = query_strategy

        if content is None and self.embeddings is None:
            raise ValueError("Either content or embeddings must be specified!")

        if content is not None:
            if embeddings is None:
                numpy_embeddings = self.embedding_fn(content)
                self.embeddings = Embeddings(
                    numpy_embeddings
                )

        # if embeddings is not provided or if a numpy array is provided
        if self.embeddings is None or isinstance(self.embeddings, np.ndarray):
            self.embeddings = Embeddings(
                self.embeddings, embedding_fn, query_strategy, self.content
            )

        if self.metadata is not None:
            if not isinstance(self.metadata, Metadata):
                self.metadata = Metadata(self.metadata)
            if "id" not in self.metadata.columns:
                if ids is None:
                    ids = [uuid.uuid4().hex for _ in range(len(self.embeddings))]
                self.metadata.metadata["id"] = ids
        else:
            if ids is None:
                ids = [uuid.uuid4().hex for _ in range(len(self.embeddings))]
            ids_df = pd.DataFrame({"id": ids})
            self.metadata = Metadata(ids_df)

        self.assert_data_types()

    def assert_data_types(self):
        # TODO: probably improve this or remove this logic entirely
        if not isinstance(self.embeddings, Embeddings):
            raise ValueError(
                "embeddings must either be provided as a numpy array or as an Embeddings object"
            )

    def __add__(self, other: Collection) -> Collection:
        content = self.content + other.content
        embeddings = self.embeddings + other.embeddings
        metadata = self.metadata + other.metadata
        return Collection(
            content=content,
            embeddings=embeddings,
            metadata=metadata,
        )

    def append(self, other: Collection) -> Collection:
        self.content.extend(other.content)
        self.embeddings = self.embeddings.append(other.embeddings)
        self.metadata = self.metadata.append(other.metadata)
        return self

    def add(
        self,
        content: List[Any] = None,
        embeddings: Union[np.ndarray, Embeddings] = None,
        ids: Optional[List[Any]] = None,
        metadata: Optional[Union[pd.DataFrame, Metadata]] = None,
    ) -> Collection:
        other = Collection(
            content,
            embeddings,
            ids,
            metadata,
            embedding_fn=self.embedding_fn,
            query_strategy=self.query_strategy,
        )
        self.append(other)
        return self

    def __len__(self) -> int:
        return len(self.embeddings)

    @property
    def ids(self) -> List[str]:
        return list(self.metadata.metadata["id"])

    @property
    def embedding_fn(self):
        return self.embeddings.embedding_fn

    @property
    def embedding_vectors(self):
        return self.embeddings.embedding_vectors

    def index(self, key) -> Collection:
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
            content=self.content[idx : idx + 1],
            embeddings=self.embeddings.index(idx),
            metadata=self.metadata.index(idx),
            embedding_fn=self.embedding_fn,
            query_strategy=self.query_strategy,
        )

    def _getiterable(self, indices: Iterable[int]) -> Collection:
        return Collection(
            content=[self.content[idx] for idx in indices],
            embeddings=self.embeddings.index(indices),
            metadata=self.metadata.index(indices),
            embedding_fn=self.embedding_fn,
            query_strategy=self.query_strategy,
        )

    def _getslice(self, index_slice: slice) -> Collection:
        return Collection(
            content=self.content[index_slice.start : index_slice.stop],
            embeddings=self.embeddings.index(index_slice),
            metadata=self.metadata.index(index_slice),
            embedding_fn=self.embedding_fn,
            query_strategy=self.query_strategy,
        )

    def query(
        self,
        query: List[Any] = None,
        query_embedding: Optional[np.ndarray] = None,
        return_collection: bool = True,
        *args,
        **kwargs,
    ) -> Union[Collection, QueryOutput, List[Collection], List[QueryOutput]]:
        if query_embedding is None:
            # TODO: maybe explicitly call batch function here?
            query_embedding = self.embeddings.embedding_fn(query)
        query_output = self.query_strategy.query(
            query_embedding, self.embeddings.embedding_vectors, *args, **kwargs
        )

        if not return_collection:
            return query_output

        if isinstance(query_output, QueryOutput):
            # not batched
            return self.index(query_output.indices)

        if len(query_output) == 0:
            return self.index(query_output[0].indices)

        # batched calls
        collection_list = []
        for q in query_output:
            collection_list.append(self.index(q.indices))
        return collection_list

    def where_in(
        self,
        not_in: bool = False,
        **kwargs,
    ):
        if not_in:
            
        pass

    def get(
        self,
        indices: Optional[List[int]] = None,
        ids: Optional[List[str]] = None,
        where: Optional[str] = None,
    ) -> Collection:
        collection = self
        if indices is not None:
            collection = self.index(indices)
        if where is not None:
            _, indices = self.metadata.filter(
                where, return_indices=True, return_metadata=False
            )
            collection = self.index(indices)
        return collection
