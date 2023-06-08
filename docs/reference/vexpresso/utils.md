# Module vexpresso.utils

??? example "View Source"
        from __future__ import annotations

        import inspect

        import os

        from collections.abc import Iterable

        from dataclasses import dataclass, field

        from functools import reduce, wraps

        from typing import Any, Callable, List, Optional, Tuple

        import daft

        ResourceRequest = daft.resource_request.ResourceRequest

        DataType = daft.datatype.DataType

        

        # LANGCHAIN

        @dataclass

        class Document:

            """Interface for interacting with a document."""

            page_content: str

            metadata: dict = field(default_factory=dict)

        

        def batchify_args(args, batch_size):

            if isinstance(args, Iterable) and not isinstance(args, str):

                if len(args) != batch_size:

                    raise ValueError("ARG needs to be size batch size")

            else:

                args = [args for _ in range(batch_size)]

            return args

        

        def lazy(default: bool = True):

            def dec(func):

                @wraps(func)

                def wrapper(*args, lazy=default, **kwargs):

                    collection = func(*args, **kwargs)

                    if not lazy:

                        collection = collection.execute()

                    return collection

                return wrapper

            return dec

        

        def convert_args(*args):

            return [

                arg.to_pylist() if isinstance(arg, daft.series.Series) else arg for arg in args

            ]

        

        def convert_kwargs(**kwargs):

            return {

                k: kwargs[k].to_pylist()

                if isinstance(kwargs[k], daft.series.Series)

                else kwargs[k]

                for k in kwargs

            }

        

        # TODO: CHANGE TO ENUM

        DATATYPES = {"python": daft.DataType.python}

        

        class TransformationWrapper:

            def __init__(

                self,

                original_transform: Transformation = None,

                datatype: str = "python",

                init_kwargs={},

            ):

                self.original_transform = original_transform

                self.datatype = datatype

                if not inspect.isclass(self.original_transform):

                    def _decorate(function: Transformation):

                        @wraps(function)

                        def wrapped(*args, **kwargs):

                            args = convert_args(*args)

                            kwargs = convert_kwargs(**kwargs)

                            return function(*args, **kwargs)

                        wrapped.__signature__ = inspect.signature(function)

                        daft_datatype = DATATYPES.get(datatype, DATATYPES["python"])

                        _udf = daft.udf(return_dtype=daft_datatype())(wrapped)

                        _udf.__vexpresso_transform = True

                        return _udf

                    self.transform = _decorate(original_transform)

                else:

                    daft_datatype = DATATYPES.get(datatype, DATATYPES["python"])

                    @daft.udf(return_dtype=daft_datatype())

                    class _Transformation:

                        def __init__(self):

                            self._transform = original_transform(**init_kwargs)

                        def __call__(self, *args, **kwargs):

                            args = convert_args(*args)

                            kwargs = convert_kwargs(**kwargs)

                            return self._transform(*args, **kwargs)

                    _Transformation.__vexpresso_transform = True

                    _Transformation.__call__.__signature__ = inspect.signature(

                        original_transform.__call__

                    )

                    self.transform = _Transformation

        

        def transformation(

            original_function: Transformation = None,

            datatype: str = "python",

            init_kwargs={},

        ):

            wrapper = TransformationWrapper(

                original_function, datatype=datatype, init_kwargs=init_kwargs

            )

            return wrapper.transform

        

        def get_field_name_and_key(field) -> Tuple[str, str]:

            field_name = field.split(".")[0]

            keys = None

            if "." in field:

                keys = field.split(".", 1)[-1]

            return field_name, keys

        

        def deep_get(dictionary, keys=None, default=None):

            if keys is None:

                return dictionary

            if isinstance(dictionary, dict):

                return reduce(

                    lambda d, key: d.get(key, default) if isinstance(d, dict) else default,

                    keys.split("."),

                    dictionary,

                )

            return dictionary

        

        class HFHubHelper:

            def __init__(self):

                self._hf_hub = None

                try:

                    import huggingface_hub  # noqa

                    self._hf_hub = huggingface_hub

                except ImportError:

                    raise ImportError(

                        "Could not import huggingface_hub python package."

                        "Please install it with `pip install huggingface_hub`."

                    )

                self.api = self._hf_hub.HfApi()

            def create_repo(

                self, repo_id: str, token: Optional[str] = None, *args, **kwargs

            ) -> str:

                if token is None:

                    token = os.environ.get("HUGGINGFACEHUB_API_TOKEN")

                return self.api.create_repo(

                    repo_id=repo_id,

                    token=token,

                    exist_ok=True,

                    repo_type="dataset",

                    *args,

                    **kwargs,

                )

            def upload(

                self,

                repo_id: str,

                folder_path: str,

                token: Optional[str] = None,

                private: bool = True,

                *args,

                **kwargs,

            ) -> str:

                if token is None:

                    token = os.environ.get("HUGGINGFACEHUB_API_TOKEN")

                self.create_repo(repo_id, token, private=private)

                return self.api.upload_folder(

                    repo_id=repo_id,

                    folder_path=folder_path,

                    token=token,

                    repo_type="dataset",

                    *args,

                    **kwargs,

                )

            def download(

                self,

                repo_id: str,

                token: Optional[str] = None,

                local_dir: Optional[str] = None,

                *args,

                **kwargs,

            ) -> str:

                if token is None:

                    token = os.environ.get("HUGGINGFACEHUB_API_TOKEN")

                return self._hf_hub.snapshot_download(

                    repo_id=repo_id,

                    token=token,

                    local_dir=local_dir,

                    repo_type="dataset",

                    *args,

                    **kwargs,

                )

        

        Transformation = Callable[[List[Any], Any], List[Any]]

## Variables

```python3
DATATYPES
```

```python3
Transformation
```

## Functions

    
### batchify_args

```python3
def batchify_args(
    args,
    batch_size
)
```

??? example "View Source"
        def batchify_args(args, batch_size):

            if isinstance(args, Iterable) and not isinstance(args, str):

                if len(args) != batch_size:

                    raise ValueError("ARG needs to be size batch size")

            else:

                args = [args for _ in range(batch_size)]

            return args

    
### convert_args

```python3
def convert_args(
    *args
)
```

??? example "View Source"
        def convert_args(*args):

            return [

                arg.to_pylist() if isinstance(arg, daft.series.Series) else arg for arg in args

            ]

    
### convert_kwargs

```python3
def convert_kwargs(
    **kwargs
)
```

??? example "View Source"
        def convert_kwargs(**kwargs):

            return {

                k: kwargs[k].to_pylist()

                if isinstance(kwargs[k], daft.series.Series)

                else kwargs[k]

                for k in kwargs

            }

    
### deep_get

```python3
def deep_get(
    dictionary,
    keys=None,
    default=None
)
```

??? example "View Source"
        def deep_get(dictionary, keys=None, default=None):

            if keys is None:

                return dictionary

            if isinstance(dictionary, dict):

                return reduce(

                    lambda d, key: d.get(key, default) if isinstance(d, dict) else default,

                    keys.split("."),

                    dictionary,

                )

            return dictionary

    
### get_field_name_and_key

```python3
def get_field_name_and_key(
    field
) -> 'Tuple[str, str]'
```

??? example "View Source"
        def get_field_name_and_key(field) -> Tuple[str, str]:

            field_name = field.split(".")[0]

            keys = None

            if "." in field:

                keys = field.split(".", 1)[-1]

            return field_name, keys

    
### lazy

```python3
def lazy(
    default: 'bool' = True
)
```

??? example "View Source"
        def lazy(default: bool = True):

            def dec(func):

                @wraps(func)

                def wrapper(*args, lazy=default, **kwargs):

                    collection = func(*args, **kwargs)

                    if not lazy:

                        collection = collection.execute()

                    return collection

                return wrapper

            return dec

    
### transformation

```python3
def transformation(
    original_function: 'Transformation' = None,
    datatype: 'str' = 'python',
    init_kwargs={}
)
```

??? example "View Source"
        def transformation(

            original_function: Transformation = None,

            datatype: str = "python",

            init_kwargs={},

        ):

            wrapper = TransformationWrapper(

                original_function, datatype=datatype, init_kwargs=init_kwargs

            )

            return wrapper.transform

## Classes

### DataType

```python3
class DataType(
    
)
```

A Daft DataType defines the type of all the values in an Expression or DataFrame column

??? example "View Source"
        class DataType:

            """A Daft DataType defines the type of all the values in an Expression or DataFrame column"""

            _dtype: PyDataType

            def __init__(self) -> None:

                raise NotImplementedError(

                    "We do not support creating a DataType via __init__ "

                    "use a creator method like DataType.int32() or use DataType.from_arrow_type(pa_type)"

                )

            @staticmethod

            def _from_pydatatype(pydt: PyDataType) -> DataType:

                dt = DataType.__new__(DataType)

                dt._dtype = pydt

                return dt

            @classmethod

            def int8(cls) -> DataType:

                """Create an 8-bit integer DataType"""

                return cls._from_pydatatype(PyDataType.int8())

            @classmethod

            def int16(cls) -> DataType:

                """Create an 16-bit integer DataType"""

                return cls._from_pydatatype(PyDataType.int16())

            @classmethod

            def int32(cls) -> DataType:

                """Create an 32-bit integer DataType"""

                return cls._from_pydatatype(PyDataType.int32())

            @classmethod

            def int64(cls) -> DataType:

                """Create an 64-bit integer DataType"""

                return cls._from_pydatatype(PyDataType.int64())

            @classmethod

            def uint8(cls) -> DataType:

                """Create an unsigned 8-bit integer DataType"""

                return cls._from_pydatatype(PyDataType.uint8())

            @classmethod

            def uint16(cls) -> DataType:

                """Create an unsigned 16-bit integer DataType"""

                return cls._from_pydatatype(PyDataType.uint16())

            @classmethod

            def uint32(cls) -> DataType:

                """Create an unsigned 32-bit integer DataType"""

                return cls._from_pydatatype(PyDataType.uint32())

            @classmethod

            def uint64(cls) -> DataType:

                """Create an unsigned 64-bit integer DataType"""

                return cls._from_pydatatype(PyDataType.uint64())

            @classmethod

            def float32(cls) -> DataType:

                """Create a 32-bit float DataType"""

                return cls._from_pydatatype(PyDataType.float32())

            @classmethod

            def float64(cls) -> DataType:

                """Create a 64-bit float DataType"""

                return cls._from_pydatatype(PyDataType.float64())

            @classmethod

            def string(cls) -> DataType:

                """Create a String DataType: A string of UTF8 characters"""

                return cls._from_pydatatype(PyDataType.string())

            @classmethod

            def bool(cls) -> DataType:

                """Create the Boolean DataType: Either ``True`` or ``False``"""

                return cls._from_pydatatype(PyDataType.bool())

            @classmethod

            def binary(cls) -> DataType:

                """Create a Binary DataType: A string of bytes"""

                return cls._from_pydatatype(PyDataType.binary())

            @classmethod

            def null(cls) -> DataType:

                """Creates the Null DataType: Always the ``Null`` value"""

                return cls._from_pydatatype(PyDataType.null())

            @classmethod

            def date(cls) -> DataType:

                """Create a Date DataType: A date with a year, month and day"""

                return cls._from_pydatatype(PyDataType.date())

            @classmethod

            def list(cls, name: str, dtype: DataType) -> DataType:

                """Create a List DataType: Variable-length list, where each element in the list has type ``dtype``

                Args:

                    dtype: DataType of each element in the list

                """

                return cls._from_pydatatype(PyDataType.list(name, dtype._dtype))

            @classmethod

            def fixed_size_list(cls, name: str, dtype: DataType, size: int) -> DataType:

                """Create a FixedSizeList DataType: Fixed-size list, where each element in the list has type ``dtype``

                and each list has length ``size``.

                Args:

                    dtype: DataType of each element in the list

                    size: length of each list

                """

                if not isinstance(size, int) or size <= 0:

                    raise ValueError("The size for a fixed-size list must be a positive integer, but got: ", size)

                return cls._from_pydatatype(PyDataType.fixed_size_list(name, dtype._dtype, size))

            @classmethod

            def struct(cls, fields: dict[str, DataType]) -> DataType:

                """Create a Struct DataType: a nested type which has names mapped to child types

                Args:

                    fields: Nested fields of the Struct

                """

                return cls._from_pydatatype(PyDataType.struct({name: datatype._dtype for name, datatype in fields.items()}))

            @classmethod

            def extension(cls, name: str, storage_dtype: DataType, metadata: str | None = None) -> DataType:

                return cls._from_pydatatype(PyDataType.extension(name, storage_dtype._dtype, metadata))

            @classmethod

            def embedding(cls, name: str, dtype: DataType, size: int) -> DataType:

                """Create an Embedding DataType: embeddings are fixed size arrays, where each element

                in the array has a **numeric** ``dtype`` and each array has a fixed length of ``size``.

                Args:

                    dtype: DataType of each element in the list (must be numeric)

                    size: length of each list

                """

                if not isinstance(size, int) or size <= 0:

                    raise ValueError("The size for a embedding must be a positive integer, but got: ", size)

                return cls._from_pydatatype(PyDataType.embedding(name, dtype._dtype, size))

            @classmethod

            def image(

                cls, mode: str | ImageMode | None = None, height: int | None = None, width: int | None = None

            ) -> DataType:

                if isinstance(mode, str):

                    mode = ImageMode.from_mode_string(mode)

                if height is not None and width is not None:

                    if not isinstance(height, int) or height <= 0:

                        raise ValueError("Image height must be a positive integer, but got: ", height)

                    if not isinstance(width, int) or width <= 0:

                        raise ValueError("Image width must be a positive integer, but got: ", width)

                elif height is not None or width is not None:

                    raise ValueError(

                        f"Image height and width must either both be specified, or both not be specified, but got height={height}, width={width}"

                    )

                return cls._from_pydatatype(PyDataType.image(mode, height, width))

            @classmethod

            def from_arrow_type(cls, arrow_type: pa.lib.DataType) -> DataType:

                """Maps a PyArrow DataType to a Daft DataType"""

                if pa.types.is_int8(arrow_type):

                    return cls.int8()

                elif pa.types.is_int16(arrow_type):

                    return cls.int16()

                elif pa.types.is_int32(arrow_type):

                    return cls.int32()

                elif pa.types.is_int64(arrow_type):

                    return cls.int64()

                elif pa.types.is_uint8(arrow_type):

                    return cls.uint8()

                elif pa.types.is_uint16(arrow_type):

                    return cls.uint16()

                elif pa.types.is_uint32(arrow_type):

                    return cls.uint32()

                elif pa.types.is_uint64(arrow_type):

                    return cls.uint64()

                elif pa.types.is_float32(arrow_type):

                    return cls.float32()

                elif pa.types.is_float64(arrow_type):

                    return cls.float64()

                elif pa.types.is_string(arrow_type) or pa.types.is_large_string(arrow_type):

                    return cls.string()

                elif pa.types.is_binary(arrow_type) or pa.types.is_large_binary(arrow_type):

                    return cls.binary()

                elif pa.types.is_boolean(arrow_type):

                    return cls.bool()

                elif pa.types.is_null(arrow_type):

                    return cls.null()

                elif pa.types.is_date32(arrow_type):

                    return cls.date()

                elif pa.types.is_list(arrow_type) or pa.types.is_large_list(arrow_type):

                    assert isinstance(arrow_type, (pa.ListType, pa.LargeListType))

                    field = arrow_type.value_field

                    return cls.list(field.name, cls.from_arrow_type(field.type))

                elif pa.types.is_fixed_size_list(arrow_type):

                    assert isinstance(arrow_type, pa.FixedSizeListType)

                    field = arrow_type.value_field

                    return cls.fixed_size_list(field.name, cls.from_arrow_type(field.type), arrow_type.list_size)

                elif pa.types.is_struct(arrow_type):

                    assert isinstance(arrow_type, pa.StructType)

                    fields = [arrow_type[i] for i in range(arrow_type.num_fields)]

                    return cls.struct({field.name: cls.from_arrow_type(field.type) for field in fields})

                elif _RAY_DATA_EXTENSIONS_AVAILABLE and isinstance(arrow_type, tuple(_TENSOR_EXTENSION_TYPES)):

                    # TODO(Clark): Add a native cross-lang extension type representation for Ray's tensor extension types.

                    return cls.python()

                elif isinstance(arrow_type, pa.PyExtensionType):

                    # TODO(Clark): Add a native cross-lang extension type representation for PyExtensionTypes.

                    raise ValueError(

                        "pyarrow extension types that subclass pa.PyExtensionType can't be used in Daft, since they can't be "

                        f"used in non-Python Arrow implementations and Daft uses the Rust Arrow2 implementation: {arrow_type}"

                    )

                elif isinstance(arrow_type, pa.BaseExtensionType):

                    name = arrow_type.extension_name

                    if (get_context().runner_config.name == "ray") and (

                        type(arrow_type).__reduce__ == pa.BaseExtensionType.__reduce__

                    ):

                        raise ValueError(

                            f"You are attempting to use a Extension Type: {arrow_type} with the default pyarrow `__reduce__` which breaks pickling for Extensions"

                            "To fix this, implement your own `__reduce__` on your extension type"

                            "For more details see this issue: "

                            "https://github.com/apache/arrow/issues/35599"

                        )

                    try:

                        metadata = arrow_type.__arrow_ext_serialize__().decode()

                    except AttributeError:

                        metadata = None

                    return cls.extension(

                        name,

                        cls.from_arrow_type(arrow_type.storage_type),

                        metadata,

                    )

                else:

                    # Fall back to a Python object type.

                    # TODO(Clark): Add native support for remaining Arrow types.

                    return cls.python()

            @classmethod

            def from_numpy_dtype(cls, np_type) -> DataType:

                """Maps a Numpy datatype to a Daft DataType"""

                arrow_type = pa.from_numpy_dtype(np_type)

                return cls.from_arrow_type(arrow_type)

            @classmethod

            def python(cls) -> DataType:

                """Create a Python DataType: a type which refers to an arbitrary Python object"""

                return cls._from_pydatatype(PyDataType.python())

            def _is_python_type(self) -> builtins.bool:

                # NOTE: This is currently used in a few places still. We can get rid of it once these are refactored away. To be discussed.

                # 1. Visualizations - we can get rid of it if we do all our repr and repr_html logic in a Series instead of in Python

                # 2. Hypothesis test data generation - we can get rid of it if we allow for creation of Series from a Python list and DataType

                return self == DataType.python()

            def __repr__(self) -> str:

                return self._dtype.__repr__()

            def __eq__(self, other: object) -> builtins.bool:

                return isinstance(other, DataType) and self._dtype.is_equal(other._dtype)

            def __getstate__(self) -> bytes:

                return self._dtype.__getstate__()

            def __setstate__(self, state: bytes) -> None:

                self._dtype = PyDataType.__new__(PyDataType)

                self._dtype.__setstate__(state)

            def __hash__(self) -> int:

                return self._dtype.__hash__()

------

#### Static methods

    
#### binary

```python3
def binary(
    
) -> 'DataType'
```

Create a Binary DataType: A string of bytes

??? example "View Source"
            @classmethod

            def binary(cls) -> DataType:

                """Create a Binary DataType: A string of bytes"""

                return cls._from_pydatatype(PyDataType.binary())

    
#### bool

```python3
def bool(
    
) -> 'DataType'
```

Create the Boolean DataType: Either ``True`` or ``False``

??? example "View Source"
            @classmethod

            def bool(cls) -> DataType:

                """Create the Boolean DataType: Either ``True`` or ``False``"""

                return cls._from_pydatatype(PyDataType.bool())

    
#### date

```python3
def date(
    
) -> 'DataType'
```

Create a Date DataType: A date with a year, month and day

??? example "View Source"
            @classmethod

            def date(cls) -> DataType:

                """Create a Date DataType: A date with a year, month and day"""

                return cls._from_pydatatype(PyDataType.date())

    
#### embedding

```python3
def embedding(
    name: 'str',
    dtype: 'DataType',
    size: 'int'
) -> 'DataType'
```

Create an Embedding DataType: embeddings are fixed size arrays, where each element

in the array has a **numeric** ``dtype`` and each array has a fixed length of ``size``.

**Parameters:**

| Name | Type | Description | Default |
|---|---|---|---|
| dtype | None | DataType of each element in the list (must be numeric) | None |
| size | None | length of each list | None |

??? example "View Source"
            @classmethod

            def embedding(cls, name: str, dtype: DataType, size: int) -> DataType:

                """Create an Embedding DataType: embeddings are fixed size arrays, where each element

                in the array has a **numeric** ``dtype`` and each array has a fixed length of ``size``.

                Args:

                    dtype: DataType of each element in the list (must be numeric)

                    size: length of each list

                """

                if not isinstance(size, int) or size <= 0:

                    raise ValueError("The size for a embedding must be a positive integer, but got: ", size)

                return cls._from_pydatatype(PyDataType.embedding(name, dtype._dtype, size))

    
#### extension

```python3
def extension(
    name: 'str',
    storage_dtype: 'DataType',
    metadata: 'str | None' = None
) -> 'DataType'
```

??? example "View Source"
            @classmethod

            def extension(cls, name: str, storage_dtype: DataType, metadata: str | None = None) -> DataType:

                return cls._from_pydatatype(PyDataType.extension(name, storage_dtype._dtype, metadata))

    
#### fixed_size_list

```python3
def fixed_size_list(
    name: 'str',
    dtype: 'DataType',
    size: 'int'
) -> 'DataType'
```

Create a FixedSizeList DataType: Fixed-size list, where each element in the list has type ``dtype``

and each list has length ``size``.

**Parameters:**

| Name | Type | Description | Default |
|---|---|---|---|
| dtype | None | DataType of each element in the list | None |
| size | None | length of each list | None |

??? example "View Source"
            @classmethod

            def fixed_size_list(cls, name: str, dtype: DataType, size: int) -> DataType:

                """Create a FixedSizeList DataType: Fixed-size list, where each element in the list has type ``dtype``

                and each list has length ``size``.

                Args:

                    dtype: DataType of each element in the list

                    size: length of each list

                """

                if not isinstance(size, int) or size <= 0:

                    raise ValueError("The size for a fixed-size list must be a positive integer, but got: ", size)

                return cls._from_pydatatype(PyDataType.fixed_size_list(name, dtype._dtype, size))

    
#### float32

```python3
def float32(
    
) -> 'DataType'
```

Create a 32-bit float DataType

??? example "View Source"
            @classmethod

            def float32(cls) -> DataType:

                """Create a 32-bit float DataType"""

                return cls._from_pydatatype(PyDataType.float32())

    
#### float64

```python3
def float64(
    
) -> 'DataType'
```

Create a 64-bit float DataType

??? example "View Source"
            @classmethod

            def float64(cls) -> DataType:

                """Create a 64-bit float DataType"""

                return cls._from_pydatatype(PyDataType.float64())

    
#### from_arrow_type

```python3
def from_arrow_type(
    arrow_type: 'pa.lib.DataType'
) -> 'DataType'
```

Maps a PyArrow DataType to a Daft DataType

??? example "View Source"
            @classmethod

            def from_arrow_type(cls, arrow_type: pa.lib.DataType) -> DataType:

                """Maps a PyArrow DataType to a Daft DataType"""

                if pa.types.is_int8(arrow_type):

                    return cls.int8()

                elif pa.types.is_int16(arrow_type):

                    return cls.int16()

                elif pa.types.is_int32(arrow_type):

                    return cls.int32()

                elif pa.types.is_int64(arrow_type):

                    return cls.int64()

                elif pa.types.is_uint8(arrow_type):

                    return cls.uint8()

                elif pa.types.is_uint16(arrow_type):

                    return cls.uint16()

                elif pa.types.is_uint32(arrow_type):

                    return cls.uint32()

                elif pa.types.is_uint64(arrow_type):

                    return cls.uint64()

                elif pa.types.is_float32(arrow_type):

                    return cls.float32()

                elif pa.types.is_float64(arrow_type):

                    return cls.float64()

                elif pa.types.is_string(arrow_type) or pa.types.is_large_string(arrow_type):

                    return cls.string()

                elif pa.types.is_binary(arrow_type) or pa.types.is_large_binary(arrow_type):

                    return cls.binary()

                elif pa.types.is_boolean(arrow_type):

                    return cls.bool()

                elif pa.types.is_null(arrow_type):

                    return cls.null()

                elif pa.types.is_date32(arrow_type):

                    return cls.date()

                elif pa.types.is_list(arrow_type) or pa.types.is_large_list(arrow_type):

                    assert isinstance(arrow_type, (pa.ListType, pa.LargeListType))

                    field = arrow_type.value_field

                    return cls.list(field.name, cls.from_arrow_type(field.type))

                elif pa.types.is_fixed_size_list(arrow_type):

                    assert isinstance(arrow_type, pa.FixedSizeListType)

                    field = arrow_type.value_field

                    return cls.fixed_size_list(field.name, cls.from_arrow_type(field.type), arrow_type.list_size)

                elif pa.types.is_struct(arrow_type):

                    assert isinstance(arrow_type, pa.StructType)

                    fields = [arrow_type[i] for i in range(arrow_type.num_fields)]

                    return cls.struct({field.name: cls.from_arrow_type(field.type) for field in fields})

                elif _RAY_DATA_EXTENSIONS_AVAILABLE and isinstance(arrow_type, tuple(_TENSOR_EXTENSION_TYPES)):

                    # TODO(Clark): Add a native cross-lang extension type representation for Ray's tensor extension types.

                    return cls.python()

                elif isinstance(arrow_type, pa.PyExtensionType):

                    # TODO(Clark): Add a native cross-lang extension type representation for PyExtensionTypes.

                    raise ValueError(

                        "pyarrow extension types that subclass pa.PyExtensionType can't be used in Daft, since they can't be "

                        f"used in non-Python Arrow implementations and Daft uses the Rust Arrow2 implementation: {arrow_type}"

                    )

                elif isinstance(arrow_type, pa.BaseExtensionType):

                    name = arrow_type.extension_name

                    if (get_context().runner_config.name == "ray") and (

                        type(arrow_type).__reduce__ == pa.BaseExtensionType.__reduce__

                    ):

                        raise ValueError(

                            f"You are attempting to use a Extension Type: {arrow_type} with the default pyarrow `__reduce__` which breaks pickling for Extensions"

                            "To fix this, implement your own `__reduce__` on your extension type"

                            "For more details see this issue: "

                            "https://github.com/apache/arrow/issues/35599"

                        )

                    try:

                        metadata = arrow_type.__arrow_ext_serialize__().decode()

                    except AttributeError:

                        metadata = None

                    return cls.extension(

                        name,

                        cls.from_arrow_type(arrow_type.storage_type),

                        metadata,

                    )

                else:

                    # Fall back to a Python object type.

                    # TODO(Clark): Add native support for remaining Arrow types.

                    return cls.python()

    
#### from_numpy_dtype

```python3
def from_numpy_dtype(
    np_type
) -> 'DataType'
```

Maps a Numpy datatype to a Daft DataType

??? example "View Source"
            @classmethod

            def from_numpy_dtype(cls, np_type) -> DataType:

                """Maps a Numpy datatype to a Daft DataType"""

                arrow_type = pa.from_numpy_dtype(np_type)

                return cls.from_arrow_type(arrow_type)

    
#### image

```python3
def image(
    mode: 'str | ImageMode | None' = None,
    height: 'int | None' = None,
    width: 'int | None' = None
) -> 'DataType'
```

??? example "View Source"
            @classmethod

            def image(

                cls, mode: str | ImageMode | None = None, height: int | None = None, width: int | None = None

            ) -> DataType:

                if isinstance(mode, str):

                    mode = ImageMode.from_mode_string(mode)

                if height is not None and width is not None:

                    if not isinstance(height, int) or height <= 0:

                        raise ValueError("Image height must be a positive integer, but got: ", height)

                    if not isinstance(width, int) or width <= 0:

                        raise ValueError("Image width must be a positive integer, but got: ", width)

                elif height is not None or width is not None:

                    raise ValueError(

                        f"Image height and width must either both be specified, or both not be specified, but got height={height}, width={width}"

                    )

                return cls._from_pydatatype(PyDataType.image(mode, height, width))

    
#### int16

```python3
def int16(
    
) -> 'DataType'
```

Create an 16-bit integer DataType

??? example "View Source"
            @classmethod

            def int16(cls) -> DataType:

                """Create an 16-bit integer DataType"""

                return cls._from_pydatatype(PyDataType.int16())

    
#### int32

```python3
def int32(
    
) -> 'DataType'
```

Create an 32-bit integer DataType

??? example "View Source"
            @classmethod

            def int32(cls) -> DataType:

                """Create an 32-bit integer DataType"""

                return cls._from_pydatatype(PyDataType.int32())

    
#### int64

```python3
def int64(
    
) -> 'DataType'
```

Create an 64-bit integer DataType

??? example "View Source"
            @classmethod

            def int64(cls) -> DataType:

                """Create an 64-bit integer DataType"""

                return cls._from_pydatatype(PyDataType.int64())

    
#### int8

```python3
def int8(
    
) -> 'DataType'
```

Create an 8-bit integer DataType

??? example "View Source"
            @classmethod

            def int8(cls) -> DataType:

                """Create an 8-bit integer DataType"""

                return cls._from_pydatatype(PyDataType.int8())

    
#### list

```python3
def list(
    name: 'str',
    dtype: 'DataType'
) -> 'DataType'
```

Create a List DataType: Variable-length list, where each element in the list has type ``dtype``

**Parameters:**

| Name | Type | Description | Default |
|---|---|---|---|
| dtype | None | DataType of each element in the list | None |

??? example "View Source"
            @classmethod

            def list(cls, name: str, dtype: DataType) -> DataType:

                """Create a List DataType: Variable-length list, where each element in the list has type ``dtype``

                Args:

                    dtype: DataType of each element in the list

                """

                return cls._from_pydatatype(PyDataType.list(name, dtype._dtype))

    
#### null

```python3
def null(
    
) -> 'DataType'
```

Creates the Null DataType: Always the ``Null`` value

??? example "View Source"
            @classmethod

            def null(cls) -> DataType:

                """Creates the Null DataType: Always the ``Null`` value"""

                return cls._from_pydatatype(PyDataType.null())

    
#### python

```python3
def python(
    
) -> 'DataType'
```

Create a Python DataType: a type which refers to an arbitrary Python object

??? example "View Source"
            @classmethod

            def python(cls) -> DataType:

                """Create a Python DataType: a type which refers to an arbitrary Python object"""

                return cls._from_pydatatype(PyDataType.python())

    
#### string

```python3
def string(
    
) -> 'DataType'
```

Create a String DataType: A string of UTF8 characters

??? example "View Source"
            @classmethod

            def string(cls) -> DataType:

                """Create a String DataType: A string of UTF8 characters"""

                return cls._from_pydatatype(PyDataType.string())

    
#### struct

```python3
def struct(
    fields: 'dict[str, DataType]'
) -> 'DataType'
```

Create a Struct DataType: a nested type which has names mapped to child types

**Parameters:**

| Name | Type | Description | Default |
|---|---|---|---|
| fields | None | Nested fields of the Struct | None |

??? example "View Source"
            @classmethod

            def struct(cls, fields: dict[str, DataType]) -> DataType:

                """Create a Struct DataType: a nested type which has names mapped to child types

                Args:

                    fields: Nested fields of the Struct

                """

                return cls._from_pydatatype(PyDataType.struct({name: datatype._dtype for name, datatype in fields.items()}))

    
#### uint16

```python3
def uint16(
    
) -> 'DataType'
```

Create an unsigned 16-bit integer DataType

??? example "View Source"
            @classmethod

            def uint16(cls) -> DataType:

                """Create an unsigned 16-bit integer DataType"""

                return cls._from_pydatatype(PyDataType.uint16())

    
#### uint32

```python3
def uint32(
    
) -> 'DataType'
```

Create an unsigned 32-bit integer DataType

??? example "View Source"
            @classmethod

            def uint32(cls) -> DataType:

                """Create an unsigned 32-bit integer DataType"""

                return cls._from_pydatatype(PyDataType.uint32())

    
#### uint64

```python3
def uint64(
    
) -> 'DataType'
```

Create an unsigned 64-bit integer DataType

??? example "View Source"
            @classmethod

            def uint64(cls) -> DataType:

                """Create an unsigned 64-bit integer DataType"""

                return cls._from_pydatatype(PyDataType.uint64())

    
#### uint8

```python3
def uint8(
    
) -> 'DataType'
```

Create an unsigned 8-bit integer DataType

??? example "View Source"
            @classmethod

            def uint8(cls) -> DataType:

                """Create an unsigned 8-bit integer DataType"""

                return cls._from_pydatatype(PyDataType.uint8())

### Document

```python3
class Document(
    page_content: 'str',
    metadata: 'dict' = <factory>
)
```

Interface for interacting with a document.

??? example "View Source"
        @dataclass

        class Document:

            """Interface for interacting with a document."""

            page_content: str

            metadata: dict = field(default_factory=dict)

------

### HFHubHelper

```python3
class HFHubHelper(
    
)
```

??? example "View Source"
        class HFHubHelper:

            def __init__(self):

                self._hf_hub = None

                try:

                    import huggingface_hub  # noqa

                    self._hf_hub = huggingface_hub

                except ImportError:

                    raise ImportError(

                        "Could not import huggingface_hub python package."

                        "Please install it with `pip install huggingface_hub`."

                    )

                self.api = self._hf_hub.HfApi()

            def create_repo(

                self, repo_id: str, token: Optional[str] = None, *args, **kwargs

            ) -> str:

                if token is None:

                    token = os.environ.get("HUGGINGFACEHUB_API_TOKEN")

                return self.api.create_repo(

                    repo_id=repo_id,

                    token=token,

                    exist_ok=True,

                    repo_type="dataset",

                    *args,

                    **kwargs,

                )

            def upload(

                self,

                repo_id: str,

                folder_path: str,

                token: Optional[str] = None,

                private: bool = True,

                *args,

                **kwargs,

            ) -> str:

                if token is None:

                    token = os.environ.get("HUGGINGFACEHUB_API_TOKEN")

                self.create_repo(repo_id, token, private=private)

                return self.api.upload_folder(

                    repo_id=repo_id,

                    folder_path=folder_path,

                    token=token,

                    repo_type="dataset",

                    *args,

                    **kwargs,

                )

            def download(

                self,

                repo_id: str,

                token: Optional[str] = None,

                local_dir: Optional[str] = None,

                *args,

                **kwargs,

            ) -> str:

                if token is None:

                    token = os.environ.get("HUGGINGFACEHUB_API_TOKEN")

                return self._hf_hub.snapshot_download(

                    repo_id=repo_id,

                    token=token,

                    local_dir=local_dir,

                    repo_type="dataset",

                    *args,

                    **kwargs,

                )

------

#### Methods

    
#### create_repo

```python3
def create_repo(
    self,
    repo_id: 'str',
    token: 'Optional[str]' = None,
    *args,
    **kwargs
) -> 'str'
```

??? example "View Source"
            def create_repo(

                self, repo_id: str, token: Optional[str] = None, *args, **kwargs

            ) -> str:

                if token is None:

                    token = os.environ.get("HUGGINGFACEHUB_API_TOKEN")

                return self.api.create_repo(

                    repo_id=repo_id,

                    token=token,

                    exist_ok=True,

                    repo_type="dataset",

                    *args,

                    **kwargs,

                )

    
#### download

```python3
def download(
    self,
    repo_id: 'str',
    token: 'Optional[str]' = None,
    local_dir: 'Optional[str]' = None,
    *args,
    **kwargs
) -> 'str'
```

??? example "View Source"
            def download(

                self,

                repo_id: str,

                token: Optional[str] = None,

                local_dir: Optional[str] = None,

                *args,

                **kwargs,

            ) -> str:

                if token is None:

                    token = os.environ.get("HUGGINGFACEHUB_API_TOKEN")

                return self._hf_hub.snapshot_download(

                    repo_id=repo_id,

                    token=token,

                    local_dir=local_dir,

                    repo_type="dataset",

                    *args,

                    **kwargs,

                )

    
#### upload

```python3
def upload(
    self,
    repo_id: 'str',
    folder_path: 'str',
    token: 'Optional[str]' = None,
    private: 'bool' = True,
    *args,
    **kwargs
) -> 'str'
```

??? example "View Source"
            def upload(

                self,

                repo_id: str,

                folder_path: str,

                token: Optional[str] = None,

                private: bool = True,

                *args,

                **kwargs,

            ) -> str:

                if token is None:

                    token = os.environ.get("HUGGINGFACEHUB_API_TOKEN")

                self.create_repo(repo_id, token, private=private)

                return self.api.upload_folder(

                    repo_id=repo_id,

                    folder_path=folder_path,

                    token=token,

                    repo_type="dataset",

                    *args,

                    **kwargs,

                )

### ResourceRequest

```python3
class ResourceRequest(
    num_cpus: 'int | float | None' = None,
    num_gpus: 'int | float | None' = None,
    memory_bytes: 'int | float | None' = None
)
```

ResourceRequest(num_cpus: 'int | float | None' = None, num_gpus: 'int | float | None' = None, memory_bytes: 'int | float | None' = None)

??? example "View Source"
        @dataclasses.dataclass(frozen=True)

        class ResourceRequest:

            num_cpus: int | float | None = None

            num_gpus: int | float | None = None

            memory_bytes: int | float | None = None

            @staticmethod

            def max_resources(resource_requests: list[ResourceRequest]) -> ResourceRequest:

                """Gets the maximum of all resources in a list of ResourceRequests as a new ResourceRequest"""

                return functools.reduce(

                    lambda acc, req: acc._max_for_each_resource(req),

                    resource_requests,

                    ResourceRequest(num_cpus=None, num_gpus=None, memory_bytes=None),

                )

            def _max_for_each_resource(self, other: ResourceRequest) -> ResourceRequest:

                """Get a new ResourceRequest that consists of the maximum requests for each resource"""

                resource_names = [f.name for f in dataclasses.fields(ResourceRequest)]

                max_resources = {}

                for name in resource_names:

                    if getattr(self, name) is None:

                        max_resources[name] = getattr(other, name)

                    elif getattr(other, name) is None:

                        max_resources[name] = getattr(self, name)

                    else:

                        max_resources[name] = max(getattr(self, name), getattr(other, name))

                return ResourceRequest(**max_resources)

            def __add__(self, other: ResourceRequest) -> ResourceRequest:

                return ResourceRequest(

                    num_cpus=add_optional_numeric(self.num_cpus, other.num_cpus),

                    num_gpus=add_optional_numeric(self.num_gpus, other.num_gpus),

                    memory_bytes=add_optional_numeric(self.memory_bytes, other.memory_bytes),

                )

------

#### Class variables

```python3
memory_bytes
```

```python3
num_cpus
```

```python3
num_gpus
```

#### Static methods

    
#### max_resources

```python3
def max_resources(
    resource_requests: 'list[ResourceRequest]'
) -> 'ResourceRequest'
```

Gets the maximum of all resources in a list of ResourceRequests as a new ResourceRequest

??? example "View Source"
            @staticmethod

            def max_resources(resource_requests: list[ResourceRequest]) -> ResourceRequest:

                """Gets the maximum of all resources in a list of ResourceRequests as a new ResourceRequest"""

                return functools.reduce(

                    lambda acc, req: acc._max_for_each_resource(req),

                    resource_requests,

                    ResourceRequest(num_cpus=None, num_gpus=None, memory_bytes=None),

                )

### TransformationWrapper

```python3
class TransformationWrapper(
    original_transform: 'Transformation' = None,
    datatype: 'str' = 'python',
    init_kwargs={}
)
```

??? example "View Source"
        class TransformationWrapper:

            def __init__(

                self,

                original_transform: Transformation = None,

                datatype: str = "python",

                init_kwargs={},

            ):

                self.original_transform = original_transform

                self.datatype = datatype

                if not inspect.isclass(self.original_transform):

                    def _decorate(function: Transformation):

                        @wraps(function)

                        def wrapped(*args, **kwargs):

                            args = convert_args(*args)

                            kwargs = convert_kwargs(**kwargs)

                            return function(*args, **kwargs)

                        wrapped.__signature__ = inspect.signature(function)

                        daft_datatype = DATATYPES.get(datatype, DATATYPES["python"])

                        _udf = daft.udf(return_dtype=daft_datatype())(wrapped)

                        _udf.__vexpresso_transform = True

                        return _udf

                    self.transform = _decorate(original_transform)

                else:

                    daft_datatype = DATATYPES.get(datatype, DATATYPES["python"])

                    @daft.udf(return_dtype=daft_datatype())

                    class _Transformation:

                        def __init__(self):

                            self._transform = original_transform(**init_kwargs)

                        def __call__(self, *args, **kwargs):

                            args = convert_args(*args)

                            kwargs = convert_kwargs(**kwargs)

                            return self._transform(*args, **kwargs)

                    _Transformation.__vexpresso_transform = True

                    _Transformation.__call__.__signature__ = inspect.signature(

                        original_transform.__call__

                    )

                    self.transform = _Transformation

------