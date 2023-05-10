from __future__ import annotations

from typing import Optional, Union, Dict, Any, Iterable
import pandas as pd
import duckdb
from duckdb import DuckDBPyConnection


# TODO: change this so we can have different storage backends. Right now we only have pandas
class Metadata:

    def __init__(
        self,
        metadata: pd.DataFrame,
        con: Optional[DuckDBPyConnection] = None
    ):
        self.metadata = metadata
        self.con = con
        if con is None:
            self.con = duckdb.connect()

    def filter(self, condition: str, return_metadata: bool = True) -> Union[Metadata, pd.DataFrame]:
        rel = self.con.from_df(self.metadata)
        df = rel.filter(condition).df()
        if return_metadata:
            return Metadata(df, self.con)
        return df

    def add(self, metadata: Union[Dict[str, Any], Iterable[Dict[str, Any]], pd.DataFrame]):
        if isinstance(metadata, dict):
            # formatting for dataframe
            metadata = [metadata]
        if not isinstance(metadata, pd.DataFrame):
            metadata = pd.DataFrame(metadata)
        self.metadata = pd.concat([self.metadata, metadata], ignore_index=True)
