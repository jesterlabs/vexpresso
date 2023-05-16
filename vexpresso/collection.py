from __future__ import annotations

import os
import uuid
from dataclasses import dataclass
from typing import Any, Callable, Dict, Iterable, List, Optional, Union

import cloudpickle
import pandas as pd

import vexpresso  # noqa
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
        embeddings: Optional[Union[Any, Embeddings]] = None,
        metadata: Optional[Union[pd.DataFrame, Dict[str, Any], Metadata]] = None,
        ids: Optional[List[int]] = None,
        embedding_fn: Union[EmbeddingFunction, Callable[[Any], Any], Any] = None,
        retrieval_strategy: RetrievalStrategy = TopKRetrievalStrategy(),
    ):
        self.content = content
        self.embeddings = embeddings
        self.metadata = metadata
        self.embedding_fn = embedding_fn
        self.retrieval_strategy = retrieval_strategy

        if not isinstance(self.embedding_fn, EmbeddingFunction):
            self.embedding_fn = EmbeddingFunction(self.embedding_fn)

        if self.content is None and self.embeddings is None:
            raise ValueError("Either content or embeddings must be specified!")

        if self.content is not None:
            if embeddings is None:
                raw_embeddings = self.embedding_fn.batch_embed(content)
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
            if "vexpresso_id" not in self.metadata.columns:
                if ids is None:
                    ids = [uuid.uuid4().hex for _ in range(len(self.embeddings))]
                self.metadata.metadata["vexpresso_id"] = ids
        else:
            if ids is None:
                ids = [uuid.uuid4().hex for _ in range(len(self.embeddings))]
            ids_df = pd.DataFrame({"vexpresso_id": ids})
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
        return self.metadata.get(["vexpresso_id"])[0]

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
        query: Any = None,
        query_embedding: Optional[Any] = None,
        return_collection: bool = True,
        *args,
        **kwargs,
    ) -> Union[Collection, QueryOutput]:
        if query is not None:
            query = [query]
        out = self.batch_query(
            query, query_embedding, return_collection, *args, **kwargs
        )
        return out[0]

    def batch_query(
        self,
        query: List[Any] = None,
        query_embedding: Optional[Union[Any, Embeddings]] = None,
        return_collection: bool = True,
        *args,
        **kwargs,
    ) -> Union[List[Collection], List[QueryOutput]]:
        if query_embedding is None:
            query_embedding = Embeddings(self.embedding_fn.batch_embed(query))

        raw_query_embedding = query_embedding

        if isinstance(query_embedding, Embeddings):
            raw_query_embedding = query_embedding.raw_embeddings

        retrieval_output = self.retrieval_strategy.retrieve(
            raw_query_embedding, self.embeddings.raw_embeddings, *args, **kwargs
        )

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
        query_kwargs: Dict[str, Any] = {},
        **kwargs,
    ) -> Collection:
        _, indices = self.metadata.where(
            column=column,
            values=values,
            return_indices=True,
            return_metadata=False,
            not_in=not_in,
            query_kwargs=query_kwargs,
            **kwargs,
        )
        return self.index(indices)

    def filter(
        self,
        condition: str,
        query_kwargs: Dict[str, Any] = {},
        **kwargs,
    ) -> Collection:
        _, indices = self.metadata.filter(
            condition,
            return_indices=True,
            return_metadata=False,
            query_kwargs=query_kwargs,
            **kwargs,
        )
        return self.index(indices)

    def save(self, path: str):
        os.makedirs(path, exist_ok=True)
        with open(os.path.join(path, "content.pkl"), mode="wb") as file:
            cloudpickle.dump(self.content, file)
        self.embeddings.save(path)
        self.metadata.save(path)

    def load(self, path: str):
        with open(os.path.join(path, "content.pkl"), mode="rb") as file:
            self.content = cloudpickle.load(file)
        self.embeddings.load(path)
        self.metadata.load(path)

    @classmethod
    def from_saved(
        cls,
        path,
        embedding_kwargs: Dict[str, Any] = {},
        metadata_kwargs: Dict[str, Any] = {},
        *args,
        **kwargs,
    ) -> Collection:
        with open(os.path.join(path, "content.pkl"), mode="rb") as file:
            content = cloudpickle.load(file)
        embeddings = Embeddings(saved_path=path)
        metadata = Metadata(saved_path=path)
        return cls(
            content=content, embeddings=embeddings, metadata=metadata, *args, **kwargs
        )
