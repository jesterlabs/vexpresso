# Module vexpresso.collection

??? example "View Source"
        from vexpresso.collection.collection import Collection

        from vexpresso.collection.daft import DaftCollection, Transformation, transformation

        __all__ = ["Collection", "DaftCollection", "Transformation", "transformation"]

## Sub-modules

* [vexpresso.collection.collection](collection/)
* [vexpresso.collection.daft](daft/)

## Variables

```python3
Transformation
```

## Functions

    
### transformation

```python3
def transformation(
    original_function: 'Transformation' = None,
    *,
    datatype: 'str' = 'python'
)
```

??? example "View Source"
        def transformation(

            original_function: Transformation = None, *, datatype: str = "python"

        ):

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

            if original_function:

                return _decorate(original_function)

            return _decorate

## Classes

### Collection

```python3
class Collection(
    /,
    *args,
    **kwargs
)
```

??? example "View Source"
        class Collection(metaclass=abc.ABCMeta):

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

            def query(

                self,

                query: Dict[str, Any],

                query_embeddings: Dict[str, Any] = {},

                filter_conditions: Optional[Dict[str, Dict[str, str]]] = None,

                *args,

                **kwargs,

            ) -> Collection:

                """

                Query method, takes in queries or query embeddings and retrieves nearest content

                Args:

                    query (Dict[str, Any]): _description_

                    query_embeddings (Dict[str, Any], optional): _description_. Defaults to {}.

                    filter_conditions (Dict[str, Dict[str, str]]): _description_

                """

            @abc.abstractmethod

            def select(self, columns: List[str]) -> Collection:

                """

                Select method, selects columns

                Returns:

                    Collection

                """

            @abc.abstractmethod

            def filter(

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

            def apply(

                self,

                column: str,

                transform_fn: Transformation,

                to: str,

                *args,

                **kwargs,

            ) -> Collection:

                """

                Apply method, takes in *args and *kwargs columns and applies a transformation function on them. The transformed columns are in format:

                transformed_{column_name}

                """

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

------

#### Descendants

* vexpresso.collection.DaftCollection

#### Static methods

    
#### from_saved

```python3
def from_saved(
    directory_or_repo_id: 'Optional[str]' = None,
    token: 'Optional[str]' = None,
    local_dir: 'Optional[str]' = None,
    to_tmpdir: 'bool' = False,
    hf_username: 'Optional[str]' = None,
    repo_name: 'Optional[str]' = None,
    hub_download_kwargs: 'Optional[Dict[str, Any]]' = {},
    *args,
    **kwargs
) -> 'Collection'
```

??? example "View Source"
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

    
#### load

```python3
def load(
    *args,
    **kwargs
) -> 'Collection'
```

??? example "View Source"
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

#### Methods

    
#### apply

```python3
def apply(
    self,
    column: 'str',
    transform_fn: 'Transformation',
    to: 'str',
    *args,
    **kwargs
) -> 'Collection'
```

Apply method, takes in *args and *kwargs columns and applies a transformation function on them. The transformed columns are in format:

transformed_{column_name}

??? example "View Source"
            @abc.abstractmethod

            def apply(

                self,

                column: str,

                transform_fn: Transformation,

                to: str,

                *args,

                **kwargs,

            ) -> Collection:

                """

                Apply method, takes in *args and *kwargs columns and applies a transformation function on them. The transformed columns are in format:

                transformed_{column_name}

                """

    
#### collect

```python3
def collect(
    self
) -> 'Collection'
```

Materializes the collection

**Returns:**

| Type | Description |
|---|---|
| Collection | Materialized collection |

??? example "View Source"
            def collect(self) -> Collection:

                """

                Materializes the collection

                Returns:

                    Collection: Materialized collection

                """

                return self

    
#### filter

```python3
def filter(
    self,
    filter_conditions: 'Dict[str, Dict[str, str]]',
    *args,
    **kwargs
) -> 'Collection'
```

Filter method, filters using conditions based on metadata

**Parameters:**

| Name | Type | Description | Default |
|---|---|---|---|
| filter_conditions | Dict[str, Dict[str, str]] | _description_ | None |

**Returns:**

| Type | Description |
|---|---|
| Collection | _description_ |

??? example "View Source"
            @abc.abstractmethod

            def filter(

                self, filter_conditions: Dict[str, Dict[str, str]], *args, **kwargs

            ) -> Collection:

                """

                Filter method, filters using conditions based on metadata

                Args:

                    filter_conditions (Dict[str, Dict[str, str]]): _description_

                Returns:

                    Collection: _description_

                """

    
#### query

```python3
def query(
    self,
    query: 'Dict[str, Any]',
    query_embeddings: 'Dict[str, Any]' = {},
    filter_conditions: 'Optional[Dict[str, Dict[str, str]]]' = None,
    *args,
    **kwargs
) -> 'Collection'
```

Query method, takes in queries or query embeddings and retrieves nearest content

**Parameters:**

| Name | Type | Description | Default |
|---|---|---|---|
| query | Dict[str, Any] | _description_ | None |
| query_embeddings | Dict[str, Any] | _description_. Defaults to {}. | {} |
| filter_conditions | Dict[str, Dict[str, str]] | _description_ | None |

??? example "View Source"
            @abc.abstractmethod

            def query(

                self,

                query: Dict[str, Any],

                query_embeddings: Dict[str, Any] = {},

                filter_conditions: Optional[Dict[str, Dict[str, str]]] = None,

                *args,

                **kwargs,

            ) -> Collection:

                """

                Query method, takes in queries or query embeddings and retrieves nearest content

                Args:

                    query (Dict[str, Any]): _description_

                    query_embeddings (Dict[str, Any], optional): _description_. Defaults to {}.

                    filter_conditions (Dict[str, Dict[str, str]]): _description_

                """

    
#### save

```python3
def save(
    self,
    directory_or_repo_id: 'Optional[str]' = None,
    to_hub: 'bool' = False,
    token: 'Optional[str]' = None,
    private: 'bool' = True,
    hf_username: 'Optional[str]' = None,
    repo_name: 'Optional[str]' = None,
    hub_kwargs: 'Optional[Dict[str, Any]]' = {}
) -> 'str'
```

??? example "View Source"
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

    
#### select

```python3
def select(
    self,
    columns: 'List[str]'
) -> 'Collection'
```

Select method, selects columns

**Returns:**

| Type | Description |
|---|---|
| None | Collection |

??? example "View Source"
            @abc.abstractmethod

            def select(self, columns: List[str]) -> Collection:

                """

                Select method, selects columns

                Returns:

                    Collection

                """

    
#### to_dict

```python3
def to_dict(
    self
) -> 'Dict[str, List[Any]]'
```

Converts collection to dict

**Returns:**

| Type | Description |
|---|---|
| Dict[str, List[Any]] | collection as dict |

??? example "View Source"
            @abc.abstractmethod

            def to_dict(self) -> Dict[str, List[Any]]:

                """

                Converts collection to dict

                Returns:

                    Dict[str, List[Any]]: collection as dict

                """

    
#### to_list

```python3
def to_list(
    self
) -> 'List[Any]'
```

Converts collection to list

**Returns:**

| Type | Description |
|---|---|
| List[Any] | returns list of columns |

??? example "View Source"
            @abc.abstractmethod

            def to_list(self) -> List[Any]:

                """

                Converts collection to list

                Returns:

                    List[Any]: returns list of columns

                """

    
#### to_pandas

```python3
def to_pandas(
    self
) -> 'pd.DataFrame'
```

Converts collection to pandas dataframe

**Returns:**

| Type | Description |
|---|---|
| pd.DataFrame | _description_ |

??? example "View Source"
            @abc.abstractmethod

            def to_pandas(self) -> pd.DataFrame:

                """

                Converts collection to pandas dataframe

                Returns:

                    pd.DataFrame: _description_

                """

### DaftCollection

```python3
class DaftCollection(
    data: 'Optional[Union[str, pd.DataFrame]]' = None,
    retriever: 'Retriever' = <vexpresso.retriever.np.NumpyRetriever object at 0x7f0c17ce6c40>,
    embedding_functions: 'Dict[str, Any]' = {},
    daft_df: 'Optional[daft.DataFrame]' = None
)
```

??? example "View Source"
        class DaftCollection(Collection):

            def __init__(

                self,

                data: Optional[Union[str, pd.DataFrame]] = None,

                retriever: Retriever = NumpyRetriever(),

                embedding_functions: Dict[str, Any] = {},

                daft_df: Optional[daft.DataFrame] = None,

            ):

                self.df = daft_df

                self.retriever = retriever

                self.embedding_functions = embedding_functions

                _metadata_dict = {}

                if data is not None:

                    if isinstance(data, str):

                        if data.endswith(".json"):

                            with open(data, "r") as f:

                                data = pd.DataFrame(json.load(f))

                    _metadata_dict = data.to_dict("list")

                if daft_df is None:

                    self.df = daft.from_pydict({**_metadata_dict})

                    self.df = self.df.with_column(

                        "vexpresso_index", indices(col(self.column_names[0]))

                    )

            def __len__(self) -> int:

                return self.df.count_rows()

            def __getitem__(self, column: str) -> Collection:

                return self.select(column)

            def __setitem__(self, column: str, value: List[Any]) -> None:

                self.df = self.add_column(column=value, name=column).df

            def set_embedding_function(self, column: str, embedding_function: Transformation):

                self.embedding_functions[column] = embedding_function

            @property

            def column_names(self) -> List[str]:

                return self.df.column_names

            def from_df(self, df: daft.DataFrame) -> DaftCollection:

                return DaftCollection(

                    retriever=self.retriever,

                    embedding_functions=self.embedding_functions,

                    daft_df=df,

                )

            def add_column(self, column: List[Any], name: str = None) -> DaftCollection:

                if name is None:

                    num_columns = len(self.df.column_names)

                    name = f"column_{num_columns}"

                new_df = daft.from_pydict({name: column})

                df = self.df.with_column(name, new_df[name])

                return self.from_df(df)

            def collect(self, in_place: bool = False):

                if in_place:

                    self.df = self.df.collect()

                    return self

                return self.from_df(self.df.collect())

            def execute(self) -> DaftCollection:

                return self.collect()

            @classmethod

            def from_collection(cls, collection: DaftCollection, **kwargs) -> DaftCollection:

                kwargs = {

                    "daft_df": collection.df,

                    "retriever": collection.retriever,

                    **kwargs,

                }

                return DaftCollection(**kwargs)

            def clone(self, **kwargs) -> DaftCollection:

                kwargs = {

                    "df": self.df,

                    "embeddings_fn": self.embeddings_fn,

                    "retriever": self.retriever,

                    **kwargs,

                }

                return DaftCollection(**kwargs)

            def to_pandas(self) -> pd.DataFrame:

                collection = self.execute()

                return collection.df.to_pandas()

            def to_dict(self) -> Dict[str, List[Any]]:

                collection = self.execute()

                return collection.df.to_pydict()

            def to_list(self) -> List[Any]:

                collection = self.execute()

                return list(collection.df.to_pydict().values())

            def show(self, num_rows: int):

                return self.df.show(num_rows)

            def _retrieve(

                self,

                df,

                column_name: str,

                query: Union[str, List[Any]],

                query_embeddings=None,

                k: int = None,

                sort=True,

                embedding_fn: Optional[Transformation] = None,

                score_column_name: Optional[str] = None,

                *args,

                **kwargs,

            ) -> daft.DataFrame:

                if embedding_fn is None:

                    embedding_fn = self.embedding_functions[column_name]

                else:

                    if column_name in self.embedding_functions:

                        if embedding_fn != self.embedding_functions[column_name]:

                            print("embedding_fn may not be the same as whats in map!")

                    else:

                        self.embedding_functions[column_name] = embedding_fn

                if query_embeddings is None:

                    query_embeddings = self.embedding_functions[column_name].func(

                        query, *args, **kwargs

                    )

                embedding_column_name = column_name

                if embedding_column_name not in df.column_names:

                    raise ValueError(

                        f"{embedding_column_name} not found in daft df. Make sure to call `embed` on column {column_name}..."

                    )

                if score_column_name is None:

                    score_column_name = f"{column_name}_score"

                df = df.with_column(

                    "retrieve_output",

                    _retrieve(

                        col(embedding_column_name),

                        query_embeddings=query_embeddings,

                        k=k,

                        retriever=self.retriever,

                    ),

                )

                df = (

                    df.with_column(

                        "retrieve_index",

                        col("retrieve_output").apply(

                            lambda x: x["retrieve_index"], return_dtype=daft.DataType.int64()

                        ),

                    )

                    .with_column(

                        score_column_name,

                        col("retrieve_output").apply(

                            lambda x: x["retrieve_score"], return_dtype=daft.DataType.float64()

                        ),

                    )

                    .exclude("retrieve_output")

                    .where(col("retrieve_index") != -1)

                    .exclude("retrieve_index")

                )

                if sort:

                    df = df.sort(col(score_column_name), desc=True)

                return df

            @lazy(default=True)

            def sort(self, column, desc=True) -> DaftCollection:

                return self.from_df(self.df.sort(col(column), desc=desc))

            @lazy(default=True)

            def query(

                self,

                column: str,

                query: List[Any] = None,

                query_embeddings: Any = None,

                filter_conditions: Optional[Dict[str, Dict[str, str]]] = None,

                k=None,

                sort=True,

                embedding_fn: Optional[Transformation] = None,

                score_column_name: Optional[str] = None,

                *args,

                **kwargs,

            ) -> DaftCollection:

                df = self.df

                if k is None:

                    k = len(self)

                df = self._retrieve(

                    df=df,

                    column_name=column,

                    query=query,

                    query_embeddings=query_embeddings,

                    k=k,

                    sort=sort,

                    embedding_fn=embedding_fn,

                    score_column_name=score_column_name,

                    *args,

                    **kwargs,

                )

                if filter_conditions is not None:

                    df = FilterHelper.filter(df, filter_conditions)

                return self.from_df(df)

            @lazy(default=True)

            def select(

                self,

                *args,

            ) -> DaftCollection:

                return self.from_df(FilterHelper.select(self.df, *args))

            @lazy(default=True)

            def filter(

                self, filter_conditions: Dict[str, Dict[str, str]], *args, **kwargs

            ) -> DaftCollection:

                return self.from_df(

                    FilterHelper.filter(self.df, filter_conditions, *args, **kwargs)

                )

            @lazy(default=True)

            def apply(

                self, transform_fn: Transformation, *args, to: Optional[str] = None, **kwargs

            ) -> DaftCollection:

                if getattr(transform_fn, "__vexpresso_transform", None) is None:

                    transform_fn = transformation(transform_fn)

                if not isinstance(args[0], DaftCollection):

                    raise TypeError(

                        "first args in apply must be a DaftCollection! use `collection['column_name']`"

                    )

                _args = []

                for _arg in args:

                    if isinstance(_arg, DaftCollection):

                        column = _arg.df.columns[0]

                        _args.append(column)

                    else:

                        _args.append(_arg)

                _kwargs = {}

                for k in kwargs:

                    _kwargs[k] = kwargs[k]

                    if isinstance(_kwargs[k], DaftCollection):

                        # only support first column

                        column = _kwargs[k].df.columns[0]

                        _kwargs[k] = column

                if to is None:

                    to = f"tranformed_{_args[0].name()}"

                df = self.df.with_column(to, transform_fn(*_args, **_kwargs))

                return self.from_df(df)

            @lazy(default=True)

            def embed(

                self,

                column_name: str,

                content: Optional[List[Any]] = None,

                embedding_fn: Optional[Transformation] = None,

                update_embedding_fn: bool = True,

                to: Optional[str] = None,

                *args,

                **kwargs,

            ) -> DaftCollection:

                collection = self

                if to is None:

                    to = f"embeddings_{column_name}"

                if content is None and column_name is None:

                    raise ValueError("column_name or content must be specified!")

                if content is not None:

                    collection = self.add_column(content, column_name)

                if embedding_fn is None:

                    embedding_fn = self.embedding_functions[to]

                else:

                    if to in self.embedding_functions:

                        if embedding_fn != self.embedding_functions[to]:

                            print("embedding_fn may not be the same as whats in map!")

                        if update_embedding_fn:

                            self.embedding_functions[to] = embedding_fn

                    else:

                        self.embedding_functions[to] = embedding_fn

                if getattr(self.embedding_functions[to], "__vexpresso_transform", None) is None:

                    self.embedding_functions[to] = transformation(self.embedding_functions[to])

                args = [self[column_name], *args]

                return collection.apply(

                    self.embedding_functions[to],

                    *args,

                    to=to,

                    **kwargs,

                )

            def save_local(self, directory: str) -> str:

                os.makedirs(directory, exist_ok=True)

                table = self.df.to_arrow()

                pq.write_table(table, os.path.join(directory, "content.parquet"))

            @classmethod

            def from_local_dir(cls, local_dir: str, *args, **kwargs) -> DaftCollection:

                df = daft.read_parquet(os.path.join(local_dir, "content.parquet"))

                return DaftCollection(daft_df=df, *args, **kwargs)

------

#### Ancestors (in MRO)

* vexpresso.collection.Collection

#### Static methods

    
#### from_collection

```python3
def from_collection(
    collection: 'DaftCollection',
    **kwargs
) -> 'DaftCollection'
```

??? example "View Source"
            @classmethod

            def from_collection(cls, collection: DaftCollection, **kwargs) -> DaftCollection:

                kwargs = {

                    "daft_df": collection.df,

                    "retriever": collection.retriever,

                    **kwargs,

                }

                return DaftCollection(**kwargs)

    
#### from_local_dir

```python3
def from_local_dir(
    local_dir: 'str',
    *args,
    **kwargs
) -> 'DaftCollection'
```

??? example "View Source"
            @classmethod

            def from_local_dir(cls, local_dir: str, *args, **kwargs) -> DaftCollection:

                df = daft.read_parquet(os.path.join(local_dir, "content.parquet"))

                return DaftCollection(daft_df=df, *args, **kwargs)

    
#### from_saved

```python3
def from_saved(
    directory_or_repo_id: 'Optional[str]' = None,
    token: 'Optional[str]' = None,
    local_dir: 'Optional[str]' = None,
    to_tmpdir: 'bool' = False,
    hf_username: 'Optional[str]' = None,
    repo_name: 'Optional[str]' = None,
    hub_download_kwargs: 'Optional[Dict[str, Any]]' = {},
    *args,
    **kwargs
) -> 'Collection'
```

??? example "View Source"
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

    
#### load

```python3
def load(
    *args,
    **kwargs
) -> 'Collection'
```

??? example "View Source"
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

#### Instance variables

```python3
column_names
```

#### Methods

    
#### add_column

```python3
def add_column(
    self,
    column: 'List[Any]',
    name: 'str' = None
) -> 'DaftCollection'
```

??? example "View Source"
            def add_column(self, column: List[Any], name: str = None) -> DaftCollection:

                if name is None:

                    num_columns = len(self.df.column_names)

                    name = f"column_{num_columns}"

                new_df = daft.from_pydict({name: column})

                df = self.df.with_column(name, new_df[name])

                return self.from_df(df)

    
#### apply

```python3
def apply(
    self,
    transform_fn: 'Transformation',
    *args,
    to: 'Optional[str]' = None,
    **kwargs
) -> 'DaftCollection'
```

Apply method, takes in *args and *kwargs columns and applies a transformation function on them. The transformed columns are in format:

transformed_{column_name}

??? example "View Source"
            @lazy(default=True)

            def apply(

                self, transform_fn: Transformation, *args, to: Optional[str] = None, **kwargs

            ) -> DaftCollection:

                if getattr(transform_fn, "__vexpresso_transform", None) is None:

                    transform_fn = transformation(transform_fn)

                if not isinstance(args[0], DaftCollection):

                    raise TypeError(

                        "first args in apply must be a DaftCollection! use `collection['column_name']`"

                    )

                _args = []

                for _arg in args:

                    if isinstance(_arg, DaftCollection):

                        column = _arg.df.columns[0]

                        _args.append(column)

                    else:

                        _args.append(_arg)

                _kwargs = {}

                for k in kwargs:

                    _kwargs[k] = kwargs[k]

                    if isinstance(_kwargs[k], DaftCollection):

                        # only support first column

                        column = _kwargs[k].df.columns[0]

                        _kwargs[k] = column

                if to is None:

                    to = f"tranformed_{_args[0].name()}"

                df = self.df.with_column(to, transform_fn(*_args, **_kwargs))

                return self.from_df(df)

    
#### clone

```python3
def clone(
    self,
    **kwargs
) -> 'DaftCollection'
```

??? example "View Source"
            def clone(self, **kwargs) -> DaftCollection:

                kwargs = {

                    "df": self.df,

                    "embeddings_fn": self.embeddings_fn,

                    "retriever": self.retriever,

                    **kwargs,

                }

                return DaftCollection(**kwargs)

    
#### collect

```python3
def collect(
    self,
    in_place: 'bool' = False
)
```

Materializes the collection

**Returns:**

| Type | Description |
|---|---|
| Collection | Materialized collection |

??? example "View Source"
            def collect(self, in_place: bool = False):

                if in_place:

                    self.df = self.df.collect()

                    return self

                return self.from_df(self.df.collect())

    
#### embed

```python3
def embed(
    self,
    column_name: 'str',
    content: 'Optional[List[Any]]' = None,
    embedding_fn: 'Optional[Transformation]' = None,
    update_embedding_fn: 'bool' = True,
    to: 'Optional[str]' = None,
    *args,
    **kwargs
) -> 'DaftCollection'
```

??? example "View Source"
            @lazy(default=True)

            def embed(

                self,

                column_name: str,

                content: Optional[List[Any]] = None,

                embedding_fn: Optional[Transformation] = None,

                update_embedding_fn: bool = True,

                to: Optional[str] = None,

                *args,

                **kwargs,

            ) -> DaftCollection:

                collection = self

                if to is None:

                    to = f"embeddings_{column_name}"

                if content is None and column_name is None:

                    raise ValueError("column_name or content must be specified!")

                if content is not None:

                    collection = self.add_column(content, column_name)

                if embedding_fn is None:

                    embedding_fn = self.embedding_functions[to]

                else:

                    if to in self.embedding_functions:

                        if embedding_fn != self.embedding_functions[to]:

                            print("embedding_fn may not be the same as whats in map!")

                        if update_embedding_fn:

                            self.embedding_functions[to] = embedding_fn

                    else:

                        self.embedding_functions[to] = embedding_fn

                if getattr(self.embedding_functions[to], "__vexpresso_transform", None) is None:

                    self.embedding_functions[to] = transformation(self.embedding_functions[to])

                args = [self[column_name], *args]

                return collection.apply(

                    self.embedding_functions[to],

                    *args,

                    to=to,

                    **kwargs,

                )

    
#### execute

```python3
def execute(
    self
) -> 'DaftCollection'
```

??? example "View Source"
            def execute(self) -> DaftCollection:

                return self.collect()

    
#### filter

```python3
def filter(
    self,
    filter_conditions: 'Dict[str, Dict[str, str]]',
    *args,
    **kwargs
) -> 'DaftCollection'
```

Filter method, filters using conditions based on metadata

**Parameters:**

| Name | Type | Description | Default |
|---|---|---|---|
| filter_conditions | Dict[str, Dict[str, str]] | _description_ | None |

**Returns:**

| Type | Description |
|---|---|
| Collection | _description_ |

??? example "View Source"
            @lazy(default=True)

            def filter(

                self, filter_conditions: Dict[str, Dict[str, str]], *args, **kwargs

            ) -> DaftCollection:

                return self.from_df(

                    FilterHelper.filter(self.df, filter_conditions, *args, **kwargs)

                )

    
#### from_df

```python3
def from_df(
    self,
    df: 'daft.DataFrame'
) -> 'DaftCollection'
```

??? example "View Source"
            def from_df(self, df: daft.DataFrame) -> DaftCollection:

                return DaftCollection(

                    retriever=self.retriever,

                    embedding_functions=self.embedding_functions,

                    daft_df=df,

                )

    
#### query

```python3
def query(
    self,
    column: 'str',
    query: 'List[Any]' = None,
    query_embeddings: 'Any' = None,
    filter_conditions: 'Optional[Dict[str, Dict[str, str]]]' = None,
    k=None,
    sort=True,
    embedding_fn: 'Optional[Transformation]' = None,
    score_column_name: 'Optional[str]' = None,
    *args,
    **kwargs
) -> 'DaftCollection'
```

Query method, takes in queries or query embeddings and retrieves nearest content

**Parameters:**

| Name | Type | Description | Default |
|---|---|---|---|
| query | Dict[str, Any] | _description_ | None |
| query_embeddings | Dict[str, Any] | _description_. Defaults to {}. | {} |
| filter_conditions | Dict[str, Dict[str, str]] | _description_ | None |

??? example "View Source"
            @lazy(default=True)

            def query(

                self,

                column: str,

                query: List[Any] = None,

                query_embeddings: Any = None,

                filter_conditions: Optional[Dict[str, Dict[str, str]]] = None,

                k=None,

                sort=True,

                embedding_fn: Optional[Transformation] = None,

                score_column_name: Optional[str] = None,

                *args,

                **kwargs,

            ) -> DaftCollection:

                df = self.df

                if k is None:

                    k = len(self)

                df = self._retrieve(

                    df=df,

                    column_name=column,

                    query=query,

                    query_embeddings=query_embeddings,

                    k=k,

                    sort=sort,

                    embedding_fn=embedding_fn,

                    score_column_name=score_column_name,

                    *args,

                    **kwargs,

                )

                if filter_conditions is not None:

                    df = FilterHelper.filter(df, filter_conditions)

                return self.from_df(df)

    
#### save

```python3
def save(
    self,
    directory_or_repo_id: 'Optional[str]' = None,
    to_hub: 'bool' = False,
    token: 'Optional[str]' = None,
    private: 'bool' = True,
    hf_username: 'Optional[str]' = None,
    repo_name: 'Optional[str]' = None,
    hub_kwargs: 'Optional[Dict[str, Any]]' = {}
) -> 'str'
```

??? example "View Source"
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

    
#### save_local

```python3
def save_local(
    self,
    directory: 'str'
) -> 'str'
```

??? example "View Source"
            def save_local(self, directory: str) -> str:

                os.makedirs(directory, exist_ok=True)

                table = self.df.to_arrow()

                pq.write_table(table, os.path.join(directory, "content.parquet"))

    
#### select

```python3
def select(
    self,
    *args
) -> 'DaftCollection'
```

Select method, selects columns

**Returns:**

| Type | Description |
|---|---|
| None | Collection |

??? example "View Source"
            @lazy(default=True)

            def select(

                self,

                *args,

            ) -> DaftCollection:

                return self.from_df(FilterHelper.select(self.df, *args))

    
#### set_embedding_function

```python3
def set_embedding_function(
    self,
    column: 'str',
    embedding_function: 'Transformation'
)
```

??? example "View Source"
            def set_embedding_function(self, column: str, embedding_function: Transformation):

                self.embedding_functions[column] = embedding_function

    
#### show

```python3
def show(
    self,
    num_rows: 'int'
)
```

??? example "View Source"
            def show(self, num_rows: int):

                return self.df.show(num_rows)

    
#### sort

```python3
def sort(
    self,
    column,
    desc=True
) -> 'DaftCollection'
```

??? example "View Source"
            @lazy(default=True)

            def sort(self, column, desc=True) -> DaftCollection:

                return self.from_df(self.df.sort(col(column), desc=desc))

    
#### to_dict

```python3
def to_dict(
    self
) -> 'Dict[str, List[Any]]'
```

Converts collection to dict

**Returns:**

| Type | Description |
|---|---|
| Dict[str, List[Any]] | collection as dict |

??? example "View Source"
            def to_dict(self) -> Dict[str, List[Any]]:

                collection = self.execute()

                return collection.df.to_pydict()

    
#### to_list

```python3
def to_list(
    self
) -> 'List[Any]'
```

Converts collection to list

**Returns:**

| Type | Description |
|---|---|
| List[Any] | returns list of columns |

??? example "View Source"
            def to_list(self) -> List[Any]:

                collection = self.execute()

                return list(collection.df.to_pydict().values())

    
#### to_pandas

```python3
def to_pandas(
    self
) -> 'pd.DataFrame'
```

Converts collection to pandas dataframe

**Returns:**

| Type | Description |
|---|---|
| pd.DataFrame | _description_ |

??? example "View Source"
            def to_pandas(self) -> pd.DataFrame:

                collection = self.execute()

                return collection.df.to_pandas()