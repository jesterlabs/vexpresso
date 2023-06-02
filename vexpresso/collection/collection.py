from __future__ import annotations

import abc
import os
import tempfile
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, Iterable, List, Optional, Union

import pandas as pd

from vexpresso.utils import Column, HFHubHelper


@dataclass
class Plan:
    function: str
    args: Iterable[Any]
    kwargs: Dict[str, Any] = field(default_factory=lambda: {})


class Collection(metaclass=abc.ABCMeta):
    def __init__(self, plan: List[Plan] = [], lazy_start: bool = False):
        self.plan = plan
        self.lazy = lazy_start

    def _from_plan(self, plan: List[Plan]) -> Collection:
        return Collection(plan)

    def collect(self) -> Collection:
        """
        Materializes the collection

        Returns:
            Collection: Materialized collection
        """
        return self

    @abc.abstractmethod
    def to_pandas(self) -> pd.DataFrame:
        """
        Converts collection to pandas dataframe

        Returns:
            pd.DataFrame: _description_
        """

    @abc.abstractmethod
    def to_dict(self) -> Dict[str, List[Any]]:
        """
        Converts collection to dict

        Returns:
            Dict[str, List[Any]]: collection as dict
        """

    @abc.abstractmethod
    def to_list(self) -> List[Any]:
        """
        Converts collection to list

        Returns:
            List[Any]: returns list of columns
        """

    @abc.abstractmethod
    def execute_query(
        self,
        query: Dict[str, Any],
        query_embeddings: Dict[str, Any] = {},
        batch: bool = False,
        *args,
        **kwargs,
    ) -> Collection:
        """
        Query method, takes in queries or query embeddings and retrieves nearest content

        Args:
            query (Dict[str, Any]): _description_
            query_embeddings (Dict[str, Any], optional): _description_. Defaults to {}.
        """

    @abc.abstractmethod
    def execute_select(self, *args) -> Collection:
        """
        Select method, selects columns

        Returns:
            Collection
        """

    @abc.abstractmethod
    def execute_filter(
        self, filter_conditions: Dict[str, Dict[str, str]], *args, **kwargs
    ) -> Collection:
        """
        Filter method, filters using conditions based on metadata

        Args:
            filter_conditions (Dict[str, Dict[str, str]]): _description_

        Returns:
            Collection: _description_
        """

    @abc.abstractmethod
    def execute_transform(
        self,
        *args: Union[Column, Any],
        to: str,
        transform: Callable[[List[Any]], List[Any]],
        **kwargs,
    ) -> Collection:
        """
        Transform method, takes in *args and *kwargs columns and applies a transformation function on them. The transformed columns are in format:
        transformed_{column_name}
        """

    def query(
        self,
        query: Dict[str, Any] = {},
        query_embeddings: Dict[str, Any] = {},
        filter_conditions: Optional[Dict[str, Dict[str, str]]] = None,
        lazy: bool = True,
        return_plan: bool = False,
        *args,
        **kwargs,
    ) -> Union[Collection, Plan]:
        new_plan = self.plan + [
            Plan(
                "execute_query",
                args=args,
                kwargs={"query": query, "query_embeddings": query_embeddings, **kwargs},
            )
        ]
        if return_plan:
            return new_plan

        collection = self._from_plan(new_plan)
        if filter_conditions is not None:
            collection = collection.filter(filter_conditions, lazy=lazy)
        if lazy:
            return collection
        return collection.execute()

    def select(
        self,
        *args,
        return_plan: bool = False,
        lazy: bool = False,
    ) -> Union[Collection, Plan]:
        new_plan = self.plan + [
            Plan(
                "execute_select",
                args=args,
            )
        ]
        if return_plan:
            return new_plan

        collection = self._from_plan(new_plan)
        if lazy:
            return collection
        return collection.execute()

    def filter(
        self,
        filter_conditions: Dict[str, Dict[str, str]],
        lazy: bool = True,
        return_plan: bool = False,
        *args,
        **kwargs,
    ) -> Union[Collection, Plan]:
        new_plan = self.plan + [
            Plan(
                "execute_filter",
                args=args,
                kwargs={
                    "filter_conditions": filter_conditions,
                    **kwargs,
                },
            )
        ]
        if return_plan:
            return new_plan

        collection = self._from_plan(new_plan)
        if lazy:
            return collection
        return collection.execute()

    def transform(
        self,
        *args,
        to: str,
        transform: Callable[[List[Any]], List[Any]] = None,
        lazy: bool = True,
        return_plan: bool = False,
        **kwargs,
    ) -> Union[Collection, Plan]:
        new_plan = self.plan + [
            Plan(
                "execute_transform",
                args=args,
                kwargs={"transform": transform, "to": to, **kwargs},
            )
        ]
        if return_plan:
            return new_plan

        collection = self._from_plan(new_plan)
        if lazy:
            return collection
        return collection.execute()

    def execute_plan(self, plan: Plan) -> Collection:
        return getattr(self, plan.function)(*plan.args, **plan.kwargs)

    def execute(self) -> Collection:
        collection = self
        for p in self.plan:
            collection = collection.execute_plan(p)
        return collection.collect()

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
    def from_saved(
        cls,
        directory_or_repo_id: Optional[str] = None,
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
        return cls.from_local_dir(saved_dir, *args, **kwargs)

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
