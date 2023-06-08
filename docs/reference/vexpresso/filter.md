# Module vexpresso.filter

??? example "View Source"
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

            def eq(cls, field: str, value: Union[str, int, float]) -> Expression:

                """

                {field} equal to {value} (str, int, float)

                """

                field_name, keys = get_field_name_and_key(field)

                def _apply_fn(col_val) -> bool:

                    return deep_get(col_val, keys=keys) == value

                return col(field_name).apply(_apply_fn, return_dtype=DataType.bool())

            @classmethod

            def neq(cls, field: str, value: Union[str, int, float]) -> Expression:

                """

                {field} not equal to {value} (str, int, float)

                """

                field_name, keys = get_field_name_and_key(field)

                def _apply_fn(col_val) -> bool:

                    return deep_get(col_val, keys=keys) != value

                return col(field_name).apply(_apply_fn, return_dtype=DataType.bool())

            @classmethod

            def gt(cls, field: str, value: Union[int, float]) -> Expression:

                """

                {field} greater than {value} (int, float)

                """

                field_name, keys = get_field_name_and_key(field)

                def _apply_fn(col_val) -> bool:

                    col_val = deep_get(col_val, keys=keys)

                    if col_val is None:

                        return False

                    return col_val > value

                return col(field_name).apply(_apply_fn, return_dtype=DataType.bool())

            @classmethod

            def gte(cls, field: str, value: Union[int, float]) -> Expression:

                """

                {field} greater than or equal to {value} (int, float)

                """

                field_name, keys = get_field_name_and_key(field)

                def _apply_fn(col_val) -> bool:

                    col_val = deep_get(col_val, keys=keys)

                    if col_val is None:

                        return False

                    return col_val >= value

                return col(field_name).apply(_apply_fn, return_dtype=DataType.bool())

            @classmethod

            def lt(cls, field: str, value: Union[int, float]) -> Expression:

                """

                {field} less than {value} (int, float)

                """

                field_name, keys = get_field_name_and_key(field)

                def _apply_fn(col_val) -> bool:

                    col_val = deep_get(col_val, keys=keys)

                    if col_val is None:

                        return False

                    return col_val < value

                return col(field_name).apply(_apply_fn, return_dtype=DataType.bool())

            @classmethod

            def lte(cls, field: str, value: Union[int, float]) -> Expression:

                """

                {field} less than or equal to {value} (int, float)

                """

                field_name, keys = get_field_name_and_key(field)

                def _apply_fn(col_val) -> bool:

                    col_val = deep_get(col_val, keys=keys)

                    if col_val is None:

                        return False

                    return col_val <= value

                return col(field_name).apply(_apply_fn, return_dtype=DataType.bool())

            @classmethod

            def isin(cls, field: str, values: List[Union[str, int, float]]) -> Expression:

                """

                {field} is in list of {values} (list of str, int, or float)

                """

                field_name, keys = get_field_name_and_key(field)

                def _apply_fn(col_val) -> bool:

                    return deep_get(col_val, keys=keys) in values

                return col(field_name).apply(_apply_fn, return_dtype=DataType.bool())

            @classmethod

            def notin(cls, field: str, values: List[Union[str, int, float]]) -> Expression:

                """

                {field} not in list of {values} (list of str, int, or float)

                """

                field_name, keys = get_field_name_and_key(field)

                def _apply_fn(col_val) -> bool:

                    return deep_get(col_val, keys=keys) not in values

                return col(field_name).apply(_apply_fn, return_dtype=DataType.bool())

            @classmethod

            def contains(cls, field: str, value: Union[str, int, float]) -> Expression:

                """

                {field} (str) contains {value} (str)

                """

                field_name, keys = get_field_name_and_key(field)

                def _apply_fn(col_val) -> bool:

                    col_val = deep_get(col_val, keys=keys)

                    if col_val is None:

                        return False

                    return value in col_val

                return col(field_name).apply(_apply_fn, return_dtype=DataType.bool())

            @classmethod

            def notcontains(cls, field: str, value: Union[str, int, float]) -> Expression:

                """

                {field} (str) does not contains {value} (str)

                """

                field_name, keys = get_field_name_and_key(field)

                def _apply_fn(col_val) -> bool:

                    col_val = deep_get(col_val, keys=keys)

                    if col_val is None:

                        return False

                    return value not in col_val

                return col(field_name).apply(_apply_fn, return_dtype=DataType.bool())

            @classmethod

            def select(cls, field: str) -> Expression:

                """

                select field

                """

                field_name, keys = get_field_name_and_key(field)

                def _apply_fn(col_val) -> Any:

                    return deep_get(col_val, keys=keys)

                return col(field_name).apply(_apply_fn, return_dtype=DataType.python())

            @classmethod

            def custom(cls, field: str, function_kwargs) -> Expression:

                field_name, keys = get_field_name_and_key(field)

                if len(function_kwargs) == 1:

                    function_kwargs = (function_kwargs, {})

                function, kwargs = function_kwargs

                def _apply_fn(col_val) -> bool:

                    return function(deep_get(col_val, keys=keys), **kwargs)

                return col(field_name).apply(_apply_fn, return_dtype=DataType.bool())

        

        class FilterHelper:

            FILTER_METHODS = FilterMethods.filter_methods()

            @classmethod

            def filter(

                cls, df: daft.DataFrame, filter_conditions: Dict[str, Dict[str, str]]

            ) -> daft.DataFrame:

                filters = []

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

                                cls.FILTER_METHODS[filter_method](metadata_field, value)

                            )

                    filt = reduce(lambda a, b: a | b, metadata_condition_fields)

                    filters.append(filt)

                op: Expression = reduce(lambda a, b: a & b, filters)

                return df.where(op)

            @classmethod

            def select(cls, df: daft.DataFrame, *args) -> daft.DataFrame:

                expressions = [FilterMethods.select(c) for c in args]

                return df.select(*expressions)

## Classes

### FilterHelper

```python3
class FilterHelper(
    /,
    *args,
    **kwargs
)
```

??? example "View Source"
        class FilterHelper:

            FILTER_METHODS = FilterMethods.filter_methods()

            @classmethod

            def filter(

                cls, df: daft.DataFrame, filter_conditions: Dict[str, Dict[str, str]]

            ) -> daft.DataFrame:

                filters = []

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

                                cls.FILTER_METHODS[filter_method](metadata_field, value)

                            )

                    filt = reduce(lambda a, b: a | b, metadata_condition_fields)

                    filters.append(filt)

                op: Expression = reduce(lambda a, b: a & b, filters)

                return df.where(op)

            @classmethod

            def select(cls, df: daft.DataFrame, *args) -> daft.DataFrame:

                expressions = [FilterMethods.select(c) for c in args]

                return df.select(*expressions)

------

#### Class variables

```python3
FILTER_METHODS
```

#### Static methods

    
#### filter

```python3
def filter(
    df: daft.dataframe.dataframe.DataFrame,
    filter_conditions: Dict[str, Dict[str, str]]
) -> daft.dataframe.dataframe.DataFrame
```

??? example "View Source"
            @classmethod

            def filter(

                cls, df: daft.DataFrame, filter_conditions: Dict[str, Dict[str, str]]

            ) -> daft.DataFrame:

                filters = []

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

                                cls.FILTER_METHODS[filter_method](metadata_field, value)

                            )

                    filt = reduce(lambda a, b: a | b, metadata_condition_fields)

                    filters.append(filt)

                op: Expression = reduce(lambda a, b: a & b, filters)

                return df.where(op)

    
#### select

```python3
def select(
    df: daft.dataframe.dataframe.DataFrame,
    *args
) -> daft.dataframe.dataframe.DataFrame
```

??? example "View Source"
            @classmethod

            def select(cls, df: daft.DataFrame, *args) -> daft.DataFrame:

                expressions = [FilterMethods.select(c) for c in args]

                return df.select(*expressions)

### FilterMethods

```python3
class FilterMethods(
    /,
    *args,
    **kwargs
)
```

??? example "View Source"
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

            def eq(cls, field: str, value: Union[str, int, float]) -> Expression:

                """

                {field} equal to {value} (str, int, float)

                """

                field_name, keys = get_field_name_and_key(field)

                def _apply_fn(col_val) -> bool:

                    return deep_get(col_val, keys=keys) == value

                return col(field_name).apply(_apply_fn, return_dtype=DataType.bool())

            @classmethod

            def neq(cls, field: str, value: Union[str, int, float]) -> Expression:

                """

                {field} not equal to {value} (str, int, float)

                """

                field_name, keys = get_field_name_and_key(field)

                def _apply_fn(col_val) -> bool:

                    return deep_get(col_val, keys=keys) != value

                return col(field_name).apply(_apply_fn, return_dtype=DataType.bool())

            @classmethod

            def gt(cls, field: str, value: Union[int, float]) -> Expression:

                """

                {field} greater than {value} (int, float)

                """

                field_name, keys = get_field_name_and_key(field)

                def _apply_fn(col_val) -> bool:

                    col_val = deep_get(col_val, keys=keys)

                    if col_val is None:

                        return False

                    return col_val > value

                return col(field_name).apply(_apply_fn, return_dtype=DataType.bool())

            @classmethod

            def gte(cls, field: str, value: Union[int, float]) -> Expression:

                """

                {field} greater than or equal to {value} (int, float)

                """

                field_name, keys = get_field_name_and_key(field)

                def _apply_fn(col_val) -> bool:

                    col_val = deep_get(col_val, keys=keys)

                    if col_val is None:

                        return False

                    return col_val >= value

                return col(field_name).apply(_apply_fn, return_dtype=DataType.bool())

            @classmethod

            def lt(cls, field: str, value: Union[int, float]) -> Expression:

                """

                {field} less than {value} (int, float)

                """

                field_name, keys = get_field_name_and_key(field)

                def _apply_fn(col_val) -> bool:

                    col_val = deep_get(col_val, keys=keys)

                    if col_val is None:

                        return False

                    return col_val < value

                return col(field_name).apply(_apply_fn, return_dtype=DataType.bool())

            @classmethod

            def lte(cls, field: str, value: Union[int, float]) -> Expression:

                """

                {field} less than or equal to {value} (int, float)

                """

                field_name, keys = get_field_name_and_key(field)

                def _apply_fn(col_val) -> bool:

                    col_val = deep_get(col_val, keys=keys)

                    if col_val is None:

                        return False

                    return col_val <= value

                return col(field_name).apply(_apply_fn, return_dtype=DataType.bool())

            @classmethod

            def isin(cls, field: str, values: List[Union[str, int, float]]) -> Expression:

                """

                {field} is in list of {values} (list of str, int, or float)

                """

                field_name, keys = get_field_name_and_key(field)

                def _apply_fn(col_val) -> bool:

                    return deep_get(col_val, keys=keys) in values

                return col(field_name).apply(_apply_fn, return_dtype=DataType.bool())

            @classmethod

            def notin(cls, field: str, values: List[Union[str, int, float]]) -> Expression:

                """

                {field} not in list of {values} (list of str, int, or float)

                """

                field_name, keys = get_field_name_and_key(field)

                def _apply_fn(col_val) -> bool:

                    return deep_get(col_val, keys=keys) not in values

                return col(field_name).apply(_apply_fn, return_dtype=DataType.bool())

            @classmethod

            def contains(cls, field: str, value: Union[str, int, float]) -> Expression:

                """

                {field} (str) contains {value} (str)

                """

                field_name, keys = get_field_name_and_key(field)

                def _apply_fn(col_val) -> bool:

                    col_val = deep_get(col_val, keys=keys)

                    if col_val is None:

                        return False

                    return value in col_val

                return col(field_name).apply(_apply_fn, return_dtype=DataType.bool())

            @classmethod

            def notcontains(cls, field: str, value: Union[str, int, float]) -> Expression:

                """

                {field} (str) does not contains {value} (str)

                """

                field_name, keys = get_field_name_and_key(field)

                def _apply_fn(col_val) -> bool:

                    col_val = deep_get(col_val, keys=keys)

                    if col_val is None:

                        return False

                    return value not in col_val

                return col(field_name).apply(_apply_fn, return_dtype=DataType.bool())

            @classmethod

            def select(cls, field: str) -> Expression:

                """

                select field

                """

                field_name, keys = get_field_name_and_key(field)

                def _apply_fn(col_val) -> Any:

                    return deep_get(col_val, keys=keys)

                return col(field_name).apply(_apply_fn, return_dtype=DataType.python())

            @classmethod

            def custom(cls, field: str, function_kwargs) -> Expression:

                field_name, keys = get_field_name_and_key(field)

                if len(function_kwargs) == 1:

                    function_kwargs = (function_kwargs, {})

                function, kwargs = function_kwargs

                def _apply_fn(col_val) -> bool:

                    return function(deep_get(col_val, keys=keys), **kwargs)

                return col(field_name).apply(_apply_fn, return_dtype=DataType.bool())

------

#### Static methods

    
#### contains

```python3
def contains(
    field: str,
    value: Union[str, int, float]
) -> daft.expressions.expressions.Expression
```

{field} (str) contains {value} (str)

??? example "View Source"
            @classmethod

            def contains(cls, field: str, value: Union[str, int, float]) -> Expression:

                """

                {field} (str) contains {value} (str)

                """

                field_name, keys = get_field_name_and_key(field)

                def _apply_fn(col_val) -> bool:

                    col_val = deep_get(col_val, keys=keys)

                    if col_val is None:

                        return False

                    return value in col_val

                return col(field_name).apply(_apply_fn, return_dtype=DataType.bool())

    
#### custom

```python3
def custom(
    field: str,
    function_kwargs
) -> daft.expressions.expressions.Expression
```

??? example "View Source"
            @classmethod

            def custom(cls, field: str, function_kwargs) -> Expression:

                field_name, keys = get_field_name_and_key(field)

                if len(function_kwargs) == 1:

                    function_kwargs = (function_kwargs, {})

                function, kwargs = function_kwargs

                def _apply_fn(col_val) -> bool:

                    return function(deep_get(col_val, keys=keys), **kwargs)

                return col(field_name).apply(_apply_fn, return_dtype=DataType.bool())

    
#### eq

```python3
def eq(
    field: str,
    value: Union[str, int, float]
) -> daft.expressions.expressions.Expression
```

{field} equal to {value} (str, int, float)

??? example "View Source"
            @classmethod

            def eq(cls, field: str, value: Union[str, int, float]) -> Expression:

                """

                {field} equal to {value} (str, int, float)

                """

                field_name, keys = get_field_name_and_key(field)

                def _apply_fn(col_val) -> bool:

                    return deep_get(col_val, keys=keys) == value

                return col(field_name).apply(_apply_fn, return_dtype=DataType.bool())

    
#### filter_methods

```python3
def filter_methods(
    
) -> Dict[str, Any]
```

??? example "View Source"
            @classmethod

            def filter_methods(cls) -> Dict[str, Any]:

                NON_FILTER_METHODS = ["filter_methods", "print_filter_methods"]

                methods = {

                    m[0]: m[1]

                    for m in inspect.getmembers(cls)

                    if not m[0].startswith("_") and m[0] not in NON_FILTER_METHODS

                }

                return methods

    
#### gt

```python3
def gt(
    field: str,
    value: Union[int, float]
) -> daft.expressions.expressions.Expression
```

{field} greater than {value} (int, float)

??? example "View Source"
            @classmethod

            def gt(cls, field: str, value: Union[int, float]) -> Expression:

                """

                {field} greater than {value} (int, float)

                """

                field_name, keys = get_field_name_and_key(field)

                def _apply_fn(col_val) -> bool:

                    col_val = deep_get(col_val, keys=keys)

                    if col_val is None:

                        return False

                    return col_val > value

                return col(field_name).apply(_apply_fn, return_dtype=DataType.bool())

    
#### gte

```python3
def gte(
    field: str,
    value: Union[int, float]
) -> daft.expressions.expressions.Expression
```

{field} greater than or equal to {value} (int, float)

??? example "View Source"
            @classmethod

            def gte(cls, field: str, value: Union[int, float]) -> Expression:

                """

                {field} greater than or equal to {value} (int, float)

                """

                field_name, keys = get_field_name_and_key(field)

                def _apply_fn(col_val) -> bool:

                    col_val = deep_get(col_val, keys=keys)

                    if col_val is None:

                        return False

                    return col_val >= value

                return col(field_name).apply(_apply_fn, return_dtype=DataType.bool())

    
#### isin

```python3
def isin(
    field: str,
    values: List[Union[str, int, float]]
) -> daft.expressions.expressions.Expression
```

{field} is in list of {values} (list of str, int, or float)

??? example "View Source"
            @classmethod

            def isin(cls, field: str, values: List[Union[str, int, float]]) -> Expression:

                """

                {field} is in list of {values} (list of str, int, or float)

                """

                field_name, keys = get_field_name_and_key(field)

                def _apply_fn(col_val) -> bool:

                    return deep_get(col_val, keys=keys) in values

                return col(field_name).apply(_apply_fn, return_dtype=DataType.bool())

    
#### lt

```python3
def lt(
    field: str,
    value: Union[int, float]
) -> daft.expressions.expressions.Expression
```

{field} less than {value} (int, float)

??? example "View Source"
            @classmethod

            def lt(cls, field: str, value: Union[int, float]) -> Expression:

                """

                {field} less than {value} (int, float)

                """

                field_name, keys = get_field_name_and_key(field)

                def _apply_fn(col_val) -> bool:

                    col_val = deep_get(col_val, keys=keys)

                    if col_val is None:

                        return False

                    return col_val < value

                return col(field_name).apply(_apply_fn, return_dtype=DataType.bool())

    
#### lte

```python3
def lte(
    field: str,
    value: Union[int, float]
) -> daft.expressions.expressions.Expression
```

{field} less than or equal to {value} (int, float)

??? example "View Source"
            @classmethod

            def lte(cls, field: str, value: Union[int, float]) -> Expression:

                """

                {field} less than or equal to {value} (int, float)

                """

                field_name, keys = get_field_name_and_key(field)

                def _apply_fn(col_val) -> bool:

                    col_val = deep_get(col_val, keys=keys)

                    if col_val is None:

                        return False

                    return col_val <= value

                return col(field_name).apply(_apply_fn, return_dtype=DataType.bool())

    
#### neq

```python3
def neq(
    field: str,
    value: Union[str, int, float]
) -> daft.expressions.expressions.Expression
```

{field} not equal to {value} (str, int, float)

??? example "View Source"
            @classmethod

            def neq(cls, field: str, value: Union[str, int, float]) -> Expression:

                """

                {field} not equal to {value} (str, int, float)

                """

                field_name, keys = get_field_name_and_key(field)

                def _apply_fn(col_val) -> bool:

                    return deep_get(col_val, keys=keys) != value

                return col(field_name).apply(_apply_fn, return_dtype=DataType.bool())

    
#### notcontains

```python3
def notcontains(
    field: str,
    value: Union[str, int, float]
) -> daft.expressions.expressions.Expression
```

{field} (str) does not contains {value} (str)

??? example "View Source"
            @classmethod

            def notcontains(cls, field: str, value: Union[str, int, float]) -> Expression:

                """

                {field} (str) does not contains {value} (str)

                """

                field_name, keys = get_field_name_and_key(field)

                def _apply_fn(col_val) -> bool:

                    col_val = deep_get(col_val, keys=keys)

                    if col_val is None:

                        return False

                    return value not in col_val

                return col(field_name).apply(_apply_fn, return_dtype=DataType.bool())

    
#### notin

```python3
def notin(
    field: str,
    values: List[Union[str, int, float]]
) -> daft.expressions.expressions.Expression
```

{field} not in list of {values} (list of str, int, or float)

??? example "View Source"
            @classmethod

            def notin(cls, field: str, values: List[Union[str, int, float]]) -> Expression:

                """

                {field} not in list of {values} (list of str, int, or float)

                """

                field_name, keys = get_field_name_and_key(field)

                def _apply_fn(col_val) -> bool:

                    return deep_get(col_val, keys=keys) not in values

                return col(field_name).apply(_apply_fn, return_dtype=DataType.bool())

    
#### print_filter_methods

```python3
def print_filter_methods(
    
)
```

??? example "View Source"
            @classmethod

            def print_filter_methods(cls):

                filter_methods = cls.filter_methods()

                for method in filter_methods:

                    description = filter_methods[method].__doc__

                    print(f"{method}: {description}")

                    print("----------------------------------")

    
#### select

```python3
def select(
    field: str
) -> daft.expressions.expressions.Expression
```

select field

??? example "View Source"
            @classmethod

            def select(cls, field: str) -> Expression:

                """

                select field

                """

                field_name, keys = get_field_name_and_key(field)

                def _apply_fn(col_val) -> Any:

                    return deep_get(col_val, keys=keys)

                return col(field_name).apply(_apply_fn, return_dtype=DataType.python())