from __future__ import annotations

import warnings
from typing import Any, Dict, Iterable, List, Optional, Tuple, Union

import pandas as pd

warnings.filterwarnings("ignore", category=pd.errors.SettingWithCopyWarning)


# TODO: change this so we can have different storage backends. Right now we only have pandas
class Metadata:
    def __init__(
        self,
        metadata: Union[pd.DataFrame, Dict[str, Any]],
    ):
        self.metadata = metadata
        if isinstance(metadata, dict):
            self.metadata = pd.DataFrame(metadata)

    def df(self) -> pd.DataFrame:
        return self.metadata

    @property
    def columns(self):
        return self.metadata.columns

    def _check_arr_type(self, arr: Iterable[Any], target_type=int) -> bool:
        for a in arr:
            if not isinstance(a, target_type):
                return False
        return True

    def index(self, indices: Iterable[int]) -> Metadata:
        return Metadata(self.metadata.iloc[indices])

    def make_empty_rows(self, num_rows: int = 1) -> Dict[str, Any]:
        columns = list(self.metadata.columns)
        dicts = []
        for _ in range(num_rows):
            dicts.append({k: None for k in columns})
        return dicts

    def get(self, variables: List[str]) -> List[Any]:
        return [list(self.metadata[v]) for v in variables]

    def where(
        self,
        column: str,
        values: List[Any],
        return_metadata: bool = True,
        return_indices: bool = False,
        query_kwargs: Dict[str, Any] = {},
        not_in: bool = False,
    ) -> Union[
        Union[Metadata, pd.DataFrame], Tuple[Union[Metadata, pd.DataFrame], List[int]]
    ]:
        if not_in:
            condition = f"{column} != @_query_values"
        else:
            condition = f"{column} == @_query_values"
        return self.filter(
            condition,
            return_metadata=return_metadata,
            return_indices=return_indices,
            query_kwargs=query_kwargs,
            _query_values=values,
        )

    def filter(
        self,
        condition: str,
        return_metadata: bool = True,
        return_indices: bool = False,
        query_kwargs: Dict[str, Any] = {},
        **kwargs,
    ) -> Union[
        Union[Metadata, pd.DataFrame], Tuple[Union[Metadata, pd.DataFrame], List[int]]
    ]:
        _query_values = kwargs.get("_query_values", None)  # noqa
        # remove temp indices
        self.metadata["vexpresso_index"] = list(range(self.metadata.shape[0]))
        df = self.metadata.query(condition, **query_kwargs)
        indices = df["vexpresso_index"]

        # remove temp index
        self.metadata = self.metadata.drop(columns=["vexpresso_index"])
        df = df.drop(columns=["vexpresso_index"])

        if return_metadata:
            if return_indices:
                return Metadata(df, self.con), indices
            return Metadata(df, self.con)
        if return_indices:
            return df, indices
        return df

    def add(
        self,
        metadata: Optional[
            Union[Dict[str, Any], Iterable[Dict[str, Any]], pd.DataFrame]
        ] = None,
    ) -> Metadata:
        if metadata is None:
            metadata = self.make_empty_rows()[0]
        if isinstance(metadata, dict):
            # formatting for dataframe
            metadata = [metadata]
        if not isinstance(metadata, pd.DataFrame):
            metadata = pd.DataFrame(metadata)
        self.metadata = pd.concat([self.metadata, metadata], ignore_index=True)
        return self
