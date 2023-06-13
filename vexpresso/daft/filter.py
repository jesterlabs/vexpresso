import inspect
from functools import reduce
from typing import Any, Dict, List, Union

import daft
from daft import col
from daft.datatype import DataType
from daft.expressions import Expression

from vexpresso.utils import deep_get, get_field_name_and_key


class FilterMethods:
    @classmethod
    def filter_methods(cls) -> Dict[str, Any]:
        NON_FILTER_METHODS = ["filter_methods", "print_filter_methods"]
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
    def eq(
        cls, field: str, value: Union[str, int, float], column_names: List[str]
    ) -> Expression:
        """
        {field} equal to {value} (str, int, float)
        """

        field_name, keys = get_field_name_and_key(field, column_names)

        def _apply_fn(col_val) -> bool:
            return deep_get(col_val, keys=keys) == value

        return col(field_name).apply(_apply_fn, return_dtype=DataType.bool())

    @classmethod
    def neq(
        cls, field: str, value: Union[str, int, float], column_names: List[str]
    ) -> Expression:
        """
        {field} not equal to {value} (str, int, float)
        """
        field_name, keys = get_field_name_and_key(field, column_names)

        def _apply_fn(col_val) -> bool:
            return deep_get(col_val, keys=keys) != value

        return col(field_name).apply(_apply_fn, return_dtype=DataType.bool())

    @classmethod
    def gt(
        cls, field: str, value: Union[int, float], column_names: List[str]
    ) -> Expression:
        """
        {field} greater than {value} (int, float)
        """
        field_name, keys = get_field_name_and_key(field, column_names)

        def _apply_fn(col_val) -> bool:
            col_val = deep_get(col_val, keys=keys)
            if col_val is None:
                return False
            return col_val > value

        return col(field_name).apply(_apply_fn, return_dtype=DataType.bool())

    @classmethod
    def gte(
        cls, field: str, value: Union[int, float], column_names: List[str]
    ) -> Expression:
        """
        {field} greater than or equal to {value} (int, float)
        """
        field_name, keys = get_field_name_and_key(field, column_names)

        def _apply_fn(col_val) -> bool:
            col_val = deep_get(col_val, keys=keys)
            if col_val is None:
                return False
            return col_val >= value

        return col(field_name).apply(_apply_fn, return_dtype=DataType.bool())

    @classmethod
    def lt(
        cls, field: str, value: Union[int, float], column_names: List[str]
    ) -> Expression:
        """
        {field} less than {value} (int, float)
        """
        field_name, keys = get_field_name_and_key(field, column_names)

        def _apply_fn(col_val) -> bool:
            col_val = deep_get(col_val, keys=keys)
            if col_val is None:
                return False
            return col_val < value

        return col(field_name).apply(_apply_fn, return_dtype=DataType.bool())

    @classmethod
    def lte(
        cls, field: str, value: Union[int, float], column_names: List[str]
    ) -> Expression:
        """
        {field} less than or equal to {value} (int, float)
        """
        field_name, keys = get_field_name_and_key(field, column_names)

        def _apply_fn(col_val) -> bool:
            col_val = deep_get(col_val, keys=keys)
            if col_val is None:
                return False
            return col_val <= value

        return col(field_name).apply(_apply_fn, return_dtype=DataType.bool())

    @classmethod
    def isin(
        cls, field: str, values: List[Union[str, int, float]], column_names: List[str]
    ) -> Expression:
        """
        {field} is in list of {values} (list of str, int, or float)
        """
        field_name, keys = get_field_name_and_key(field, column_names)

        def _apply_fn(col_val) -> bool:
            return deep_get(col_val, keys=keys) in values

        return col(field_name).apply(_apply_fn, return_dtype=DataType.bool())

    @classmethod
    def notin(
        cls, field: str, values: List[Union[str, int, float]], column_names: List[str]
    ) -> Expression:
        """
        {field} not in list of {values} (list of str, int, or float)
        """
        field_name, keys = get_field_name_and_key(field, column_names)

        def _apply_fn(col_val) -> bool:
            return deep_get(col_val, keys=keys) not in values

        return col(field_name).apply(_apply_fn, return_dtype=DataType.bool())

    @classmethod
    def contains(
        cls, field: str, value: Union[str, int, float], column_names: List[str]
    ) -> Expression:
        """
        {field} (str) contains {value} (str)
        """
        field_name, keys = get_field_name_and_key(field, column_names)

        def _apply_fn(col_val) -> bool:
            col_val = deep_get(col_val, keys=keys)
            if col_val is None:
                return False
            return value in col_val

        return col(field_name).apply(_apply_fn, return_dtype=DataType.bool())

    @classmethod
    def notcontains(
        cls, field: str, value: Union[str, int, float], column_names: List[str]
    ) -> Expression:
        """
        {field} (str) does not contains {value} (str)
        """
        field_name, keys = get_field_name_and_key(field, column_names)

        def _apply_fn(col_val) -> bool:
            col_val = deep_get(col_val, keys=keys)
            if col_val is None:
                return False
            return value not in col_val

        return col(field_name).apply(_apply_fn, return_dtype=DataType.bool())

    @classmethod
    def select(cls, field: str, column_names: List[str]) -> Expression:
        """
        select field
        """
        field_name, keys = get_field_name_and_key(field, column_names)

        def _apply_fn(col_val) -> Any:
            return deep_get(col_val, keys=keys)

        return (
            col(field_name)
            .apply(_apply_fn, return_dtype=DataType.python())
            .alias(field)
        )

    @classmethod
    def custom(cls, field: str, function_kwargs, column_names: List[str]) -> Expression:
        field_name, keys = get_field_name_and_key(field, column_names)

        function = function_kwargs.get("function")
        function_kwargs = function_kwargs.get("function_kwargs", {})

        def _apply_fn(col_val) -> bool:
            return function(deep_get(col_val, keys=keys), **function_kwargs)

        return col(field_name).apply(_apply_fn, return_dtype=DataType.bool())


class FilterHelper:
    FILTER_METHODS = FilterMethods.filter_methods()

    @classmethod
    def filter(
        cls, df: daft.DataFrame, filter_conditions: Dict[str, Dict[str, str]]
    ) -> daft.DataFrame:
        """
        Filter format:
        {
            <field>: {
                <filter_method>: <value>
            },
            <field>: {
                <filter_method>: <value>
            },
        }
        """
        filters = []
        column_names = df.column_names
        for metadata_field in filter_conditions:
            metadata_conditions = filter_conditions[metadata_field]
            if isinstance(metadata_conditions, dict):
                # otherwise its a list1
                metadata_conditions = [metadata_conditions]
            metadata_condition_fields = []
            for m in metadata_conditions:
                for filter_method in m:
                    if filter_method not in cls.FILTER_METHODS:
                        raise ValueError(
                            f"""
                                filter_method: {filter_method} not in supported filter methods: {cls.FILTER_METHODS}.
                            """
                        )
                    value = m[filter_method]
                    metadata_condition_fields.append(
                        cls.FILTER_METHODS[filter_method](
                            metadata_field, value, column_names
                        )
                    )
            filt = reduce(lambda a, b: a | b, metadata_condition_fields)
            filters.append(filt)
        op: Expression = reduce(lambda a, b: a & b, filters)
        return df.where(op)

    @classmethod
    def select(cls, df: daft.DataFrame, *args) -> daft.DataFrame:
        column_names = df.column_names
        expressions = [FilterMethods.select(c, column_names) for c in args]
        return df.select(*expressions)
