from __future__ import annotations

from typing import Any, Dict, Iterable, List, Optional, Tuple, Union

import duckdb
import pandas as pd
from duckdb import DuckDBPyConnection


# TODO: change this so we can have different storage backends. Right now we only have pandas
class Metadata:
    def __init__(
        self,
        metadata: Union[pd.DataFrame, Dict[str, Any]],
        con: Optional[DuckDBPyConnection] = None,
    ):
        self.metadata = metadata
        if isinstance(metadata, dict):
            self.metadata = pd.DataFrame(metadata)
        self.con = con
        if con is None:
            self.con = duckdb.connect()

    @property
    def columns(self):
        return self.metadata.columns

    def _check_arr_type(self, arr: Iterable[Any], target_type=int) -> bool:
        for a in arr:
            if not isinstance(a, target_type):
                return False
        return True

    def index(self, key) -> Metadata:
        if isinstance(key, int):
            return self._getitemidx(key)
        elif isinstance(key, slice):
            return self._getslice(key)
        elif isinstance(key, Iterable):
            if self._check_arr_type(key, str):
                return self._getitemstr(key)
            return self._getiterable(key)
        else:
            raise TypeError(
                "Index must be int, str, or iterable of ints or strings, not {}".format(
                    type(key).__name__
                )
            )

    def _getitemidx(self, idx: int) -> Metadata:
        return Metadata(self.metadata.iloc[idx : idx + 1], self.con)

    def _getitemstr(self, columns: Iterable[str]) -> Metadata:
        return Metadata(self.metadata[columns], self.con)

    def _getiterable(self, indices: Iterable[int]) -> Metadata:
        return Metadata(self.metadata.iloc[indices], self.con)

    def _getslice(self, index_slice: slice) -> Metadata:
        return Metadata(self.metadata[index_slice.start : index_slice.stop], self.con)

    def make_empty_rows(self, num_rows: int = 1) -> Dict[str, Any]:
        columns = list(self.metadata.columns)
        dicts = []
        for _ in range(num_rows):
            dicts.append({k: None for k in columns})
        return dicts

    def where_in(
        self,
        column: str,
        values: List[Any],
        return_metadata: bool = True,
        return_indices: bool = False,
        not_in: bool = False,
    ) -> Metadata:
        # remove temp indices
        self.metadata["vexpresso_index"] = list(range(self.metadata.shape[0]))
        if not_in:
            df = self.metadata.query(f"{column} not in {values}")
        else:
            df = self.metadata.query(f"{column} in {values}")
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

    def filter(
        self, condition: str, return_metadata: bool = True, return_indices: bool = False
    ) -> Union[
        Union[Metadata, pd.DataFrame], Tuple[Union[Metadata, pd.DataFrame], List[int]]
    ]:
        # remove temp indices
        self.metadata["vexpresso_index"] = list(range(self.metadata.shape[0]))
        rel = self.con.from_df(self.metadata)
        df = rel.filter(condition).df()
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

    def __add__(self, other: Metadata) -> Metadata:
        metadata = pd.concat([self.metadata, other.metadata], ignore_index=True)
        return Metadata(metadata, self.con)

    def append(self, other: Metadata) -> Metadata:
        self.metadata = pd.concat([self.metadata, other.metadata], ignore_index=True)
        return self

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
