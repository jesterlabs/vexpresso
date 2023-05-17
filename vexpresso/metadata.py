from __future__ import annotations

import inspect
import uuid

# import os
import warnings
from typing import Any, Dict, Iterable, List, Optional, Tuple, Union

import duckdb
import pandas as pd

warnings.filterwarnings("ignore", category=pd.errors.SettingWithCopyWarning)


class FilterStringHelper:
    @classmethod
    def filter_methods(cls) -> Dict[str, Any]:
        NON_FILTER_METHODS = ["call", "filter_methods", "print_filter_methods"]
        methods = {
            m[0]: m[1]
            for m in inspect.getmembers(cls)
            if not m[0].startswith("_") and m[0] not in NON_FILTER_METHODS
        }
        return methods

    @classmethod
    def print_filter_methods(cls):
        filter_methods = cls.filter_methods()
        for method in filter_methods:
            description = filter_methods[method].__doc__
            print(f"{method}: {description}")
            print("----------------------------------")

    @classmethod
    def eq(cls, field: str, value: Union[str, int, float]) -> str:
        """
        {field} equal to {value} (str, int, float)
        """
        return f"{field} == '{value}'"

    @classmethod
    def neq(cls, field: str, value: Union[str, int, float]) -> str:
        """
        {field} not equal to {value} (str, int, float)
        """
        return f"{field} != '{value}'"

    @classmethod
    def gt(cls, field: str, value: Union[int, float]) -> str:
        """
        {field} greater than {value} (int, float)
        """
        return f"{field} > '{value}'"

    @classmethod
    def gte(cls, field: str, value: Union[int, float]) -> str:
        """
        {field} greater than or equal to {value} (int, float)
        """
        return f"{field} >= '{value}'"

    @classmethod
    def lt(cls, field: str, value: Union[int, float]) -> str:
        """
        {field} less than {value} (int, float)
        """
        return f"{field} < '{value}'"

    @classmethod
    def lte(cls, field: str, value: Union[int, float]) -> str:
        """
        {field} less than or equal to {value} (int, float)
        """
        return f"{field} <= '{value}'"

    @classmethod
    def isin(cls, field: str, values: List[Union[str, int, float]]) -> str:
        """
        {field} is in list of {values} (list of str, int, or float)
        """
        values = tuple(values)
        return f"{field} in {values}"

    @classmethod
    def notin(cls, field: str, values: List[Union[str, int, float]]) -> str:
        """
        {field} not in list of {values} (list of str, int, or float)
        """
        values = tuple(values)
        return f"{field} not in {values}"

    @classmethod
    def contains(cls, field: str, value: str) -> str:
        """
        {field} (str) contains {value} (str)
        """
        return f"{field} LIKE '%{value}%'"

    @classmethod
    def notcontains(cls, field: str, value: str) -> str:
        """
        {field} (str) does not contains {value} (str)
        """
        return f"{field} NOT LIKE '%{value}%'"


# TODO: change this so we can have different storage backends. Right now we only have pandas
class Metadata:
    def __init__(
        self,
        metadata: Union[pd.DataFrame, Dict[str, Any], str, Metadata] = None,
        ids: Optional[List[str]] = None,
        length: Optional[int] = None,
    ):
        self.metadata = metadata
        if length is None:
            length = self.__len__()
        if metadata is None:
            if length == 0:
                raise ValueError("metadata or length must be specified!")
            self.metadata = pd.DataFrame({})
        elif isinstance(metadata, pd.DataFrame):
            self.metadata = metadata
        elif isinstance(metadata, dict):
            self.metadata = pd.DataFrame(metadata)
        elif isinstance(metadata, str):
            self.load(metadata)
        elif isinstance(metadata, Metadata):
            self.metadata = metadata.metadata
        else:
            raise ValueError(
                "metadata must be either of types: (pd.DataFrame, Dict[str, Any], str, Metadata)"
            )

        self.add_fields({"vexpresso_index": list(range(length))})

        if ids is None:
            if "id" in self.fields:
                ids = self.get_field("id")
            else:
                ids = [uuid.uuid4().hex for _ in range(length)]
        self.add_fields({"id": ids})

        self.con = self.create_duckdb_con(self.metadata)

    def create_duckdb_con(self, df) -> duckdb.DuckDBPyRelation:
        return duckdb.from_df(df)

    def __len__(self) -> int:
        if self.metadata is None:
            return 0
        return self.metadata.shape[0]

    def add_fields(self, fields: Dict[str, Iterable[Any]]):
        for field in fields:
            self.metadata[field] = fields[field]
        self.con = self.create_duckdb_con(self.metadata)

    @property
    def fields(self) -> List[str]:
        return list(self.metadata.columns)

    def get_field(self, field: str) -> List[Any]:
        agg = self.con.aggregate(f"list({field})").df()
        return list(agg.iloc[0])[0]

    def get_fields(self, fields: List[str]) -> List[List[Any]]:
        _fields = [f"list({f})" for f in fields]
        field_str = ", ".join(_fields)
        agg = self.con.aggregate(field_str).df()
        return list(agg.iloc[0])

    def df(self) -> pd.DataFrame:
        return self.metadata

    def index(self, indices: Iterable[int]) -> Metadata:
        return Metadata(self.metadata.iloc[indices])

    def make_empty_rows(self, num_rows: int = 1) -> Dict[str, Any]:
        columns = list(self.metadata.columns)
        dicts = []
        for _ in range(num_rows):
            dicts.append({k: None for k in columns})
        return dicts

    @classmethod
    def filter_methods(cls) -> Dict[str, Any]:
        return FilterStringHelper.filter_methods()

    @classmethod
    def print_filter_methods(cls):
        FilterStringHelper.print_filter_methods()

    def filter(
        self,
        filter_conditions: Dict[str, Dict[str, str]] = None,
        filter_string: Optional[str] = None,
    ) -> Tuple[Metadata, List[int]]:
        filter_methods = self.filter_methods()

        if filter_conditions is None and filter_string is None:
            raise ValueError(
                "Either filter_conditions or filter_string must be specified!"
            )

        filters = []
        if filter_string is not None:
            filters.append(filter_string)

        if filter_conditions is not None:
            for metadata_field in filter_conditions:
                metadata_conditions = filter_conditions[metadata_field]
                for filter_method in metadata_conditions:
                    if filter_method not in filter_methods:
                        raise ValueError(
                            f"""
                                filter_method: {filter_method} not in supported filter methods: {filter_methods}.
                            """
                        )
                    value = metadata_conditions[filter_method]
                    filters.append(filter_methods[filter_method](metadata_field, value))
        full_filter_string = " AND ".join(filters)
        df = self.con.filter(full_filter_string).df()
        indices = list(df["vexpresso_index"])
        return Metadata(df), indices

    def add(
        self,
        metadata: Optional[
            Union[Dict[str, Any], Iterable[Dict[str, Any]], pd.DataFrame, Metadata]
        ] = None,
    ) -> Metadata:
        if metadata is None:
            metadata = self.make_empty_rows()[0]
        elif isinstance(metadata, dict):
            # formatting for dataframe
            metadata = [metadata]
            metadata = pd.DataFrame(metadata)
        elif isinstance(metadata, Metadata):
            metadata = metadata.metadata
        self.metadata = pd.concat([self.metadata, metadata], ignore_index=True)
        self.add_fields({"vexpresso_index": list(range(self.__len__()))})
        return self

    # def save(self, path: str, filename: Optional[str] = None) -> str:
    #     if filename is None:
    #         filename = "metadata.csv"
    #     path = os.path.join(path, filename)
    #     self.metadata.to_csv(path)
    #     return path

    # def load(self, path: str, filename: Optional[str] = None):
    #     if filename is None:
    #         filename = "metadata.csv"
    #     path = os.path.join(path, filename)
    #     self.metadata = pd.read_csv(path)
