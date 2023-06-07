# Module vexpresso.collection

??? example "View Source"
        from __future__ import annotations

        import abc

        import os

        import tempfile

        from typing import Any, Dict, List, Optional

        import pandas as pd

        from vexpresso.utils import HFHubHelper, Transformation, batchify_args

        

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

                column: str,

                query: Any = None,

                query_embeddings: Any = None,

                filter_conditions: Optional[Dict[str, Dict[str, str]]] = None,

                k=None,

                sort=True,

                embedding_fn: Optional[Transformation] = None,

                score_column_name: Optional[str] = None,

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

            def batch_query(

                self,

                columns: List[str],

                queries: List[Any] = None,

                query_embeddings: List[Any] = None,

                filter_conditions: List[Optional[Dict[str, Dict[str, str]]]] = None,

                k=None,

                sort=True,

                embedding_fn: List[Optional[Transformation]] = None,

                score_column_name: List[Optional[str]] = None,

                *args,

                **kwargs,

            ) -> List[Collection]:

                batch_size = len(columns)

                queries = batchify_args(queries, batch_size)

                query_embeddings = batchify_args(query_embeddings, batch_size)

                filter_conditions = batchify_args(filter_conditions, batch_size)

                k = batchify_args(k, batch_size)

                sort = batchify_args(sort, batch_size)

                embedding_fn = batchify_args(embedding_fn, batch_size)

                score_column_name = batchify_args(score_column_name, batch_size)

                collection = self

                collections = []

                for i in range(batch_size):

                    collections.append(

                        collection.query(

                            columns[i],

                            queries[i],

                            query_embeddings[i],

                            filter_conditions[i],

                            k[i],

                            sort[i],

                            embedding_fn[i],

                            score_column_name[i],

                            *args,

                            **kwargs,

                        )

                    )

                return collections

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

                self, transform_fn: Transformation, *args, to: Optional[str] = None, **kwargs

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

            @classmethod

            def connect(

                cls, address: str = None, cluster_kwargs: Dict[str, Any] = {}, *args, **kwargs

            ) -> Collection:

                return None

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

                column: str,

                query: Any = None,

                query_embeddings: Any = None,

                filter_conditions: Optional[Dict[str, Dict[str, str]]] = None,

                k=None,

                sort=True,

                embedding_fn: Optional[Transformation] = None,

                score_column_name: Optional[str] = None,

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

            def batch_query(

                self,

                columns: List[str],

                queries: List[Any] = None,

                query_embeddings: List[Any] = None,

                filter_conditions: List[Optional[Dict[str, Dict[str, str]]]] = None,

                k=None,

                sort=True,

                embedding_fn: List[Optional[Transformation]] = None,

                score_column_name: List[Optional[str]] = None,

                *args,

                **kwargs,

            ) -> List[Collection]:

                batch_size = len(columns)

                queries = batchify_args(queries, batch_size)

                query_embeddings = batchify_args(query_embeddings, batch_size)

                filter_conditions = batchify_args(filter_conditions, batch_size)

                k = batchify_args(k, batch_size)

                sort = batchify_args(sort, batch_size)

                embedding_fn = batchify_args(embedding_fn, batch_size)

                score_column_name = batchify_args(score_column_name, batch_size)

                collection = self

                collections = []

                for i in range(batch_size):

                    collections.append(

                        collection.query(

                            columns[i],

                            queries[i],

                            query_embeddings[i],

                            filter_conditions[i],

                            k[i],

                            sort[i],

                            embedding_fn[i],

                            score_column_name[i],

                            *args,

                            **kwargs,

                        )

                    )

                return collections

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

                self, transform_fn: Transformation, *args, to: Optional[str] = None, **kwargs

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

            @classmethod

            def connect(

                cls, address: str = None, cluster_kwargs: Dict[str, Any] = {}, *args, **kwargs

            ) -> Collection:

                return None

------

#### Descendants

* vexpresso.daft.collection.DaftCollection

#### Static methods

    
#### connect

```python3
def connect(
    address: 'str' = None,
    cluster_kwargs: 'Dict[str, Any]' = {},
    *args,
    **kwargs
) -> 'Collection'
```

??? example "View Source"
            @classmethod

            def connect(

                cls, address: str = None, cluster_kwargs: Dict[str, Any] = {}, *args, **kwargs

            ) -> Collection:

                return None

    
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
    transform_fn: 'Transformation',
    *args,
    to: 'Optional[str]' = None,
    **kwargs
) -> 'Collection'
```

Apply method, takes in *args and *kwargs columns and applies a transformation function on them. The transformed columns are in format:

transformed_{column_name}

??? example "View Source"
            @abc.abstractmethod

            def apply(

                self, transform_fn: Transformation, *args, to: Optional[str] = None, **kwargs

            ) -> Collection:

                """

                Apply method, takes in *args and *kwargs columns and applies a transformation function on them. The transformed columns are in format:

                transformed_{column_name}

                """

    
#### batch_query

```python3
def batch_query(
    self,
    columns: 'List[str]',
    queries: 'List[Any]' = None,
    query_embeddings: 'List[Any]' = None,
    filter_conditions: 'List[Optional[Dict[str, Dict[str, str]]]]' = None,
    k=None,
    sort=True,
    embedding_fn: 'List[Optional[Transformation]]' = None,
    score_column_name: 'List[Optional[str]]' = None,
    *args,
    **kwargs
) -> 'List[Collection]'
```

??? example "View Source"
            def batch_query(

                self,

                columns: List[str],

                queries: List[Any] = None,

                query_embeddings: List[Any] = None,

                filter_conditions: List[Optional[Dict[str, Dict[str, str]]]] = None,

                k=None,

                sort=True,

                embedding_fn: List[Optional[Transformation]] = None,

                score_column_name: List[Optional[str]] = None,

                *args,

                **kwargs,

            ) -> List[Collection]:

                batch_size = len(columns)

                queries = batchify_args(queries, batch_size)

                query_embeddings = batchify_args(query_embeddings, batch_size)

                filter_conditions = batchify_args(filter_conditions, batch_size)

                k = batchify_args(k, batch_size)

                sort = batchify_args(sort, batch_size)

                embedding_fn = batchify_args(embedding_fn, batch_size)

                score_column_name = batchify_args(score_column_name, batch_size)

                collection = self

                collections = []

                for i in range(batch_size):

                    collections.append(

                        collection.query(

                            columns[i],

                            queries[i],

                            query_embeddings[i],

                            filter_conditions[i],

                            k[i],

                            sort[i],

                            embedding_fn[i],

                            score_column_name[i],

                            *args,

                            **kwargs,

                        )

                    )

                return collections

    
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
    column: 'str',
    query: 'Any' = None,
    query_embeddings: 'Any' = None,
    filter_conditions: 'Optional[Dict[str, Dict[str, str]]]' = None,
    k=None,
    sort=True,
    embedding_fn: 'Optional[Transformation]' = None,
    score_column_name: 'Optional[str]' = None,
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

                column: str,

                query: Any = None,

                query_embeddings: Any = None,

                filter_conditions: Optional[Dict[str, Dict[str, str]]] = None,

                k=None,

                sort=True,

                embedding_fn: Optional[Transformation] = None,

                score_column_name: Optional[str] = None,

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