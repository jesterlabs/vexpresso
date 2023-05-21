from __future__ import annotations

import os
import tempfile
from dataclasses import dataclass
from typing import Any, Callable, Dict, Iterable, List, Optional, Union

import cloudpickle
import pandas as pd

import vexpresso  # noqa
from vexpresso.embedding_function import EmbeddingFunction
from vexpresso.embeddings import Embeddings
from vexpresso.hub_utils import HFHubHelper
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
        ids: Optional[Union[List[str], str]] = None,
        embeddings_fn: Union[EmbeddingFunction, Callable[[Any], Any], Any] = None,
        retrieval_strategy: RetrievalStrategy = TopKRetrievalStrategy(),
        saved_path: Optional[str] = None,
    ):
        self.content = content
        self.embeddings = embeddings
        self.metadata = metadata
        self.embeddings_fn = embeddings_fn
        self.retrieval_strategy = retrieval_strategy

        if saved_path is not None:
            self.load(saved_path)

        self._init_setup(ids)

    def _init_setup(self, ids):
        if not isinstance(self.embeddings_fn, EmbeddingFunction):
            self.embeddings_fn = EmbeddingFunction(self.embeddings_fn)

        if self.content is None and self.embeddings is None:
            raise ValueError("Either content or embeddings must be specified!")

        if isinstance(self.content, str):
            # content points to something in metadata
            self.metadata = Metadata(metadata=self.metadata)
            self.content = self.metadata.get_field(self.content)

        if self.content is not None:
            if self.embeddings is None:
                raw_embeddings = self.embeddings_fn.batch_embed(self.content)
                self.embeddings = Embeddings(raw_embeddings)
        else:
            self.content = [None for _ in range(len(self.embeddings))]

        if not isinstance(self.embeddings, Embeddings):
            raw_embeddings = self.embeddings
            self.embeddings = Embeddings(raw_embeddings)

        self.metadata = Metadata(
            metadata=self.metadata, length=len(self.embeddings), ids=ids
        )

        self.assert_data_types()

    def assert_data_types(self):
        # TODO: probably improve this or remove this logic entirely
        if not isinstance(self.embeddings, Embeddings):
            raise ValueError(
                "embeddings must either be provided as a numpy array or as an Embeddings object"
            )

    def df(self) -> pd.DataFrame:
        df = self.metadata.df()
        df["content"] = self.content
        return df

    @property
    def ids(self) -> List[Any]:
        return self.metadata.get_field("id")

    @property
    def fields(self) -> List[str]:
        return self.metadata.fields

    def get_field(self, field: str) -> List[Any]:
        return self.metadata.get_field(field)

    def get_fields(self, fields: List[str]) -> List[List[Any]]:
        return self.metadata.get_fields(fields)

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
            embeddings_fn=self.embeddings_fn,
            retrieval_strategy=self.retrieval_strategy,
        )
        content = self.content + other.content
        embeddings = self.embeddings.add(other.embeddings)
        metadata = self.metadata.add(other.metadata)
        return Collection(
            content=content,
            embeddings=embeddings,
            metadata=metadata,
            embeddings_fn=self.embeddings_fn,
            retrieval_strategy=self.retrieval_strategy,
        )

    def remove(self, ids: Union[List[str], str]) -> Collection:
        if isinstance(ids, str):
            filter_condition = {"id": {"neq": ids}}
        else:
            filter_condition = {"id": {"notcontains": ids}}
        return self.filter(filter_condition)

    def index(self, indices: Iterable[int]) -> Collection:
        return Collection(
            content=[self.content[idx] for idx in indices],
            embeddings=self.embeddings.index(indices),
            metadata=self.metadata.index(indices),
            embeddings_fn=self.embeddings_fn,
            retrieval_strategy=self.retrieval_strategy,
        )

    def query(
        self,
        query: Any = None,
        query_embedding: Optional[Any] = None,
        return_collection: bool = True,
        filter_conditions: Optional[Dict[str, Dict[str, str]]] = None,
        filter_string: Optional[str] = None,
        embeddings: Optional[Embeddings] = None,
        *args,
        **kwargs,
    ) -> Union[Collection, QueryOutput]:
        if filter_conditions is not None or filter_string is not None:
            new_collection = self.filter(filter_conditions, filter_string)
            return new_collection.query(query, query_embedding, return_collection)
        if query is not None:
            query = [query]
        out = self.batch_query(
            query,
            query_embedding,
            return_collection,
            embeddings=embeddings,
            *args,
            **kwargs,
        )
        return out[0]

    def batch_query(
        self,
        query: List[Any] = None,
        query_embedding: Optional[Union[Any, Embeddings]] = None,
        return_collection: bool = True,
        embeddings: Optional[Embeddings] = None,
        *args,
        **kwargs,
    ) -> Union[List[Collection], List[QueryOutput]]:
        if embeddings is None:
            embeddings = self.embeddings

        if query_embedding is None:
            query_embedding = Embeddings(self.embeddings_fn.batch_embed(query))

        raw_query_embedding = query_embedding

        if isinstance(query_embedding, Embeddings):
            raw_query_embedding = query_embedding.raw_embeddings

        retrieval_output = self.retrieval_strategy.retrieve(
            raw_query_embedding, embeddings.raw_embeddings, *args, **kwargs
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

    def filter(
        self,
        filter_conditions: Dict[str, Dict[str, str]] = None,
        filter_string: Optional[str] = None,
    ) -> Collection:
        _, indices = self.metadata.filter(filter_conditions, filter_string)
        return self.index(indices)

    def save_local(self, directory: str):
        os.makedirs(directory, exist_ok=True)
        with open(os.path.join(directory, "content.pkl"), mode="wb") as file:
            cloudpickle.dump(self.content, file)
        self.embeddings.save(os.path.join(directory, "embeddings.npy"))
        self.metadata.save(os.path.join(directory, "metadata.csv"))
        return directory

    def save(
        self,
        directory_or_repo_id: Optional[str] = None,
        to_hub: bool = False,
        token: Optional[str] = None,
        private: bool = True,
        hf_username: Optional[str] = None,
        repo_name: Optional[str] = None,
        hub_kwargs: Optional[Dict[str, Any]] = {},
    ) -> str:
        if to_hub:
            print(f"Uploading collection to {directory_or_repo_id}")
            if directory_or_repo_id is None:
                if hf_username is None or repo_name is None:
                    raise ValueError(
                        "Please provide either a directory / repo id or your huggingface username + repo name"
                    )
                directory_or_repo_id = f"{hf_username}/{repo_name}"
            with tempfile.TemporaryDirectory() as tmpdirname:
                self.save_local(tmpdirname)
                helper = HFHubHelper()
                helper.upload(
                    repo_id=directory_or_repo_id,
                    folder_path=tmpdirname,
                    token=token,
                    private=private,
                    **hub_kwargs,
                )
            print(f"Upload to {directory_or_repo_id} complete!")
            return directory_or_repo_id
        else:
            print(f"saving to {directory_or_repo_id}")
            return self.save_local(directory_or_repo_id)

    @classmethod
    def from_local_dir(
        cls,
        directory: str,
        embeddings_kwargs: Dict[str, Any] = {},
        metadata_kwargs: Dict[str, Any] = {},
        *args,
        **kwargs,
    ) -> Collection:
        with open(os.path.join(directory, "content.pkl"), mode="rb") as file:
            content = cloudpickle.load(file)
        embeddings = Embeddings(
            os.path.join(directory, "embeddings.npy"), **embeddings_kwargs
        )
        metadata = Metadata(os.path.join(directory, "metadata.csv"), **metadata_kwargs)
        return cls(
            content=content, embeddings=embeddings, metadata=metadata, *args, **kwargs
        )

    @classmethod
    def from_saved(
        cls,
        directory_or_repo_id: Optional[str] = None,
        embeddings_kwargs: Dict[str, Any] = {},
        metadata_kwargs: Dict[str, Any] = {},
        token: Optional[str] = None,
        local_dir: Optional[str] = None,
        to_tmpdir: bool = False,
        hf_username: Optional[str] = None,
        repo_name: Optional[str] = None,
        hub_download_kwargs: Optional[Dict[str, Any]] = {},
        *args,
        **kwargs,
    ) -> Collection:
        if directory_or_repo_id is None:
            if hf_username is None or repo_name is None:
                raise ValueError(
                    "Please provide either a directory / repo id or your huggingface username + repo name"
                )
            directory_or_repo_id = f"{hf_username}/{repo_name}"
        saved_dir = directory_or_repo_id
        if not os.path.isdir(directory_or_repo_id):
            # from huggingface
            print(f"Retrieving from hf repo: {directory_or_repo_id}")
            with tempfile.TemporaryDirectory() as tmpdirname:
                helper = HFHubHelper()
                if to_tmpdir:
                    local_dir = tmpdirname
                saved_dir = helper.download(
                    directory_or_repo_id,
                    token=token,
                    local_dir=local_dir,
                    **hub_download_kwargs,
                )
        return cls.from_local_dir(
            saved_dir, embeddings_kwargs, metadata_kwargs, *args, **kwargs
        )

    @classmethod
    def load(
        cls,
        *args,
        **kwargs,
    ) -> Collection:
        return cls.from_saved(
            *args,
            **kwargs,
        )
