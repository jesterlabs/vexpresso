from __future__ import annotations

import uuid
from dataclasses import dataclass
from typing import Any, Callable, Dict, Iterable, List, Optional, Union

import pandas as pd

from vexpresso.embedding_function import EmbeddingFunction
from vexpresso.embeddings import Embeddings
from vexpresso.metadata import Metadata
from vexpresso.retrieval import (
    RetrievalOutput,
    RetrievalStrategy,
    TopKRetrievalStrategy,
)


@dataclass
class QueryOutput:
    retrieval: RetrievalOutput
    content: Optional[Any] = None


class Collection:
    def __init__(
        self,
        content: Optional[List[Any]] = None,
        embeddings: Optional[Any] = None,
        metadata: Optional[Union[pd.DataFrame, Dict[str, Any], Metadata]] = None,
        ids: Optional[List[int]] = None,
        embedding_fn: Union[EmbeddingFunction, Callable[[Any], Any]] = None,
        retrieval_strategy: RetrievalStrategy = TopKRetrievalStrategy(),
    ):
        self.content = content
        self.embeddings = embeddings
        self.metadata = metadata
        self.embedding_fn = embedding_fn
        self.retrieval_strategy = retrieval_strategy

        if self.content is None and self.embeddings is None:
            raise ValueError("Either content or embeddings must be specified!")

        if self.content is not None:
            if embeddings is None:
                raw_embeddings = self.embedding_fn(content)
                self.embeddings = Embeddings(raw_embeddings)

        # if embeddings is not provided
        if not isinstance(self.embeddings, Embeddings):
            raw_embeddings = self.embeddings
            self.embeddings = Embeddings(raw_embeddings)

        if self.content is None:
            self.content = [None for _ in range(len(self.embeddings))]

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

    @property
    def ids(self) -> List[Any]:
        return self.metadata.get(["id"])[0]

    def __len__(self) -> int:
        """
        Returns the size of the collection

        Returns:
            int: Size of collection.
        """
        return len(self.embeddings)

    def add(
        self,
        content: List[Any] = None,
        embeddings: Optional[Any] = None,
        ids: Optional[List[str]] = None,
        metadata: Optional[Union[pd.DataFrame, Metadata, Dict[str, Any]]] = None,
    ) -> Collection:
        """Add a collection"""
        other = Collection(
            content=content,
            embeddings=embeddings,
            ids=ids,
            metadata=metadata,
            embedding_fn=self.embedding_fn,
            retrieval_strategy=self.retrieval_strategy,
        )
        return self.append(other)

    def append(self, other: Collection) -> Collection:
        self.content.extend(other.content)
        self.embeddings = self.embeddings.append(other.embeddings)
        self.metadata = self.metadata.append(other.metadata)
        return self

    def index(self, indices: Iterable[int]) -> Collection:
        return Collection(
            content=[self.content[idx] for idx in indices],
            embeddings=self.embeddings.index(indices),
            metadata=self.metadata.index(indices),
            embedding_fn=self.embedding_fn,
            retrieval_strategy=self.retrieval_strategy,
        )

    def query(
        self,
        query: List[Any] = None,
        query_embedding: Optional[Any] = None,
        return_collection: bool = True,
        *args,
        **kwargs,
    ) -> Union[Collection, QueryOutput, List[Collection], List[QueryOutput]]:
        if query_embedding is None:
            # TODO: maybe explicitly call batch function here?
            query_embedding = self.embedding_fn(query)
        retrieval_output = self.retrieval_strategy.retrieve(
            query_embedding, self.embeddings.raw_embeddings, *args, **kwargs
        )

        if isinstance(retrieval_output, RetrievalOutput):
            # not batched
            if not return_collection:
                return QueryOutput(
                    retrieval_output,
                    [self.content[idx] for idx in retrieval_output.indices],
                )
            return self.index(retrieval_output.indices)

        if len(retrieval_output) == 0:
            if not return_collection:
                return QueryOutput(
                    retrieval_output[0],
                    [self.content[idx] for idx in retrieval_output[0].indices],
                )
            return self.index(retrieval_output[0].indices)

        if not return_collection:
            query_output_list = []
            for r in retrieval_output:
                query_output = QueryOutput(r, [self.content[idx] for idx in r.indices])
                query_output_list.append(query_output)
            return query_output_list

        # batched calls
        collection_list = []
        for r in retrieval_output:
            collection_list.append(self.index(r.indices))
        return collection_list

    def get(
        self,
        indices: Optional[List[int]] = None,
        ids: Optional[List[str]] = None,
    ) -> Collection:
        collection = self
        if indices is not None:
            collection = collection.index(indices)
        if ids is not None:
            _, indices = self.metadata.where(
                column="id", values=ids, return_indices=True, return_metadata=False
            )
            collection = collection.index(indices)
        return collection

    def where(
        self,
        column: str,
        values: List[Any],
        not_in: bool = False,
    ) -> Collection:
        _, indices = self.metadata.where(
            column=column,
            values=values,
            return_indices=True,
            return_metadata=False,
            not_in=not_in,
        )
        return self.index(indices)
