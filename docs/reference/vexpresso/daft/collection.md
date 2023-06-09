# Module vexpresso.daft.collection

??? example "View Source"
        from __future__ import annotations

        import json

        import os

        from typing import Any, Dict, Iterable, List, Optional, Union

        import daft

        # import numpy as np

        import pandas as pd

        import pyarrow.parquet as pq

        import ray

        from daft import col

        from vexpresso.collection import Collection

        from vexpresso.daft.filter import FilterHelper

        from vexpresso.daft.utils import Wrapper, indices, retrieve

        from vexpresso.embeddings import get_embedding_fn

        from vexpresso.retriever import NumpyRetriever, Retriever

        from vexpresso.utils import (

            DataType,

            Document,

            ResourceRequest,

            Transformation,

            lazy,

            transformation,

        )

        

        class DaftCollection(Collection):

            def __init__(

                self,

                data: Optional[Union[str, pd.DataFrame, Dict[str, Any]]] = None,

                retriever: Retriever = NumpyRetriever(),

                embedding_functions: Dict[str, Any] = {},

                daft_df: Optional[daft.DataFrame] = None,

            ):

                self.daft_df = daft_df

                self.retriever = retriever

                self.embedding_functions = embedding_functions

                _metadata = {}

                if data is not None:

                    if isinstance(data, str):

                        if data.endswith(".json"):

                            with open(data, "r") as f:

                                _metadata = pd.DataFrame(json.load(f))

                    elif isinstance(data, pd.DataFrame):

                        _metadata = data.to_dict("list")

                    else:

                        _metadata = data

                if daft_df is None and len(_metadata) > 0:

                    if isinstance(_metadata, list):

                        self.daft_df = daft.from_pylist(_metadata)

                    else:

                        self.daft_df = daft.from_pydict({**_metadata})

                    self.daft_df = self.daft_df.with_column(

                        "vexpresso_index", indices(col(self.column_names[0]))

                    )

            @property

            def df(self) -> Wrapper:

                return Wrapper(self)

            def __len__(self) -> int:

                return self.daft_df.count_rows()

            def __getitem__(self, column: str) -> DaftCollection:

                return self.select(column)

            def __setitem__(self, column: str, value: List[Any]) -> None:

                self.daft_df = self.add_column(column=value, name=column).df

            def cast(

                self, column: str = None, datatype: DataType = DataType.python()

            ) -> DaftCollection:

                if column is None:

                    columns = [col(c).cast(datatype) for c in self.column_names]

                else:

                    columns = [col(column).cast(datatype)]

                return self.from_daft_df(self.daft_df.select(*columns))

            def add_rows(self, data: List[Dict[str, Any]]) -> DaftCollection:

                dic = self.to_dict()

                for k in dic:

                    for d in data:

                        value = d.get(k, None)

                        dic[k].append(value)

                return self.from_data(dic)

            def set_embedding_function(self, column: str, embedding_function: Transformation):

                self.embedding_functions[column] = embedding_function

            @property

            def column_names(self) -> List[str]:

                return self.daft_df.column_names

            def from_daft_df(self, df: daft.DataFrame) -> DaftCollection:

                return DaftCollection(

                    retriever=self.retriever,

                    embedding_functions=self.embedding_functions,

                    daft_df=df,

                )

            def from_data(self, data: Any) -> DaftCollection:

                return DaftCollection(

                    data=data,

                    retriever=self.retriever,

                    embedding_functions=self.embedding_functions,

                )

            def add_column(self, column: List[Any], name: str = None) -> DaftCollection:

                if name is None:

                    num_columns = len(self.daft_df.column_names)

                    name = f"column_{num_columns}"

                new_df = daft.from_pydict({name: column})

                df = self.daft_df.with_column(name, new_df[name])

                return self.from_daft_df(df)

            def collect(self, in_place: bool = False):

                if in_place:

                    self.daft_df = self.daft_df.collect()

                    return self

                return self.from_daft_df(self.daft_df.collect())

            def execute(self) -> DaftCollection:

                return self.collect()

            def to_pandas(self) -> pd.DataFrame:

                collection = self.execute()

                return collection.daft_df.to_pandas()

            def to_dict(self) -> Dict[str, List[Any]]:

                collection = self.execute()

                return collection.daft_df.to_pydict()

            def to_list(self) -> List[Any]:

                collection = self.execute()

                return list(collection.daft_df.to_pydict().values())

            def show(self, num_rows: int):

                return self.daft_df.show(num_rows)

            @lazy(default=True)

            def sort(self, column, desc=True) -> DaftCollection:

                return self.from_daft_df(self.daft_df.sort(col(column), desc=desc))

            def _embed_queries(

                self,

                queries,

                embedding_function,

                resource_request=ResourceRequest(),

                *args,

                **kwargs,

            ):

                query_embeddings = (

                    daft.from_pydict({"queries": queries})

                    .with_column(

                        "query_embeddings",

                        embedding_function(col("queries"), *args, **kwargs),

                        resource_request=resource_request,

                    )

                    .select("query_embeddings")

                    .collect()

                    .to_pydict()["query_embeddings"]

                )

                return query_embeddings

            @lazy(default=True)

            def query(

                self,

                column: str,

                query: List[Any] = None,

                query_embedding: List[Any] = None,

                filter_conditions: Optional[Dict[str, Dict[str, str]]] = None,

                k: int = None,

                sort: bool = True,

                embedding_fn: Optional[Transformation] = None,

                score_column_name: Optional[str] = None,

                resource_request: ResourceRequest = ResourceRequest(),

                *args,

                **kwargs,

            ) -> Collection:

                if query is not None:

                    query = [query]

                return self.batch_query(

                    column=column,

                    queries=query,

                    query_embeddings=query_embedding,

                    filter_conditions=filter_conditions,

                    k=k,

                    sort=sort,

                    embedding_fn=embedding_fn,

                    score_column_name=score_column_name,

                    resource_request=resource_request,

                    *args,

                    **kwargs,

                )[0]

            @lazy(default=True)

            def batch_query(

                self,

                column: str,

                queries: List[Any] = None,

                query_embeddings: List[Any] = None,

                filter_conditions: Optional[Dict[str, Dict[str, str]]] = None,

                k: int = None,

                sort: bool = True,

                embedding_fn: Optional[Transformation] = None,

                score_column_name: Optional[str] = None,

                resource_request: ResourceRequest = ResourceRequest(),

                *args,

                **kwargs,

            ) -> List[Collection]:

                batch_size = len(queries) if query_embeddings is None else len(query_embeddings)

                if embedding_fn is not None:

                    if column in self.embedding_functions:

                        if embedding_fn != self.embedding_functions[column]:

                            print(

                                "embedding_fn may not be the same as whats in map! Updating what's in map..."

                            )

                    self.embedding_functions[column] = get_embedding_fn(embedding_fn)

                if query_embeddings is None:

                    query_embeddings = self._embed_queries(

                        queries,

                        self.embedding_functions[column],

                        resource_request,

                        *args,

                        **kwargs,

                    )

                dfs = retrieve(

                    batch_size,

                    self.daft_df,

                    column,

                    query_embeddings,

                    self.retriever,

                    k,

                    sort,

                    score_column_name,

                    resource_request,

                )

                for i in range(len(dfs)):

                    if filter_conditions is not None:

                        dfs[i] = FilterHelper.filter(dfs[i], filter_conditions)

                return [self.from_daft_df(df) for df in dfs]

            @lazy(default=True)

            def select(

                self,

                *args,

            ) -> DaftCollection:

                return self.from_daft_df(FilterHelper.select(self.daft_df, *args))

            @lazy(default=True)

            def exclude(

                self,

                *args,

            ) -> DaftCollection:

                return self.from_daft_df(self.daft_df.exclude(*args))

            @lazy(default=True)

            def filter(

                self, filter_conditions: Dict[str, Dict[str, str]], *args, **kwargs

            ) -> DaftCollection:

                return self.from_daft_df(

                    FilterHelper.filter(self.daft_df, filter_conditions, *args, **kwargs)

                )

            @lazy(default=True)

            def apply(

                self,

                transform_fn: Transformation,

                *args,

                to: Optional[str] = None,

                resource_request: ResourceRequest = ResourceRequest(),

                **kwargs,

            ) -> DaftCollection:

                transform_fn = transformation(transform_fn)

                if not isinstance(args[0], DaftCollection):

                    raise TypeError(

                        "first args in apply must be a DaftCollection! use `collection['column_name']`"

                    )

                _args = []

                for _arg in args:

                    if isinstance(_arg, DaftCollection):

                        column = _arg.daft_df.columns[0]

                        _args.append(column)

                    else:

                        _args.append(_arg)

                _kwargs = {}

                for k in kwargs:

                    _kwargs[k] = kwargs[k]

                    if isinstance(_kwargs[k], DaftCollection):

                        # only support first column

                        column = _kwargs[k].daft_df.columns[0]

                        _kwargs[k] = column

                if to is None:

                    to = f"tranformed_{_args[0].name()}"

                df = self.daft_df.with_column(

                    to, transform_fn(*_args, **_kwargs), resource_request=resource_request

                )

                return self.from_daft_df(df)

            @lazy(default=True)

            def embed(

                self,

                column_name: str,

                *args,

                content: Optional[List[Any]] = None,

                embedding_fn: Optional[Transformation] = None,

                update_embedding_fn: bool = True,

                to: Optional[str] = None,

                resource_request: ResourceRequest = ResourceRequest(),

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

                self.embedding_functions[to] = get_embedding_fn(self.embedding_functions[to])

                args = [self[column_name], *args]

                return collection.apply(

                    self.embedding_functions[to],

                    *args,

                    to=to,

                    resource_request=resource_request,

                    **kwargs,

                )

            def save_local(self, directory: str) -> str:

                os.makedirs(directory, exist_ok=True)

                table = self.daft_df.to_arrow()

                pq.write_table(table, os.path.join(directory, "content.parquet"))

            @classmethod

            def from_local_dir(cls, local_dir: str, *args, **kwargs) -> DaftCollection:

                df = daft.read_parquet(os.path.join(local_dir, "content.parquet"))

                return DaftCollection(daft_df=df, *args, **kwargs)

            @classmethod

            def connect(

                cls, address: str = None, cluster_kwargs: Dict[str, Any] = {}, *args, **kwargs

            ) -> DaftCollection:

                if address is None:

                    addy = ray.init(**cluster_kwargs)

                else:

                    addy = ray.init(address=address, **cluster_kwargs)

                daft.context.set_runner_ray(address=addy.address_info["address"])

                return DaftCollection(*args, **kwargs)

            def to_langchain(self, document_column: str, embeddings_column: str):

                from langchain.docstore.document import Document

                from langchain.vectorstores import VectorStore

                class VexpressoVectorStore(VectorStore):

                    def __init__(self, collection: DaftCollection):

                        self.collection = collection

                        self.document_column = document_column

                        self.embeddings_column = embeddings_column

                    def add_texts(

                        self,

                        texts: Iterable[str],

                        metadatas: Optional[List[dict]] = None,

                        **kwargs: Any,

                    ) -> List[str]:

                        if metadatas is None:

                            metadatas = [{} for _ in range(len(texts))]

                        combined = [

                            {self.document_column: t, **m} for t, m in zip(texts, metadatas)

                        ]

                        self.collection = self.collection.add_rows(combined)

                    def similarity_search(

                        self, query: str, k: int = 4, **kwargs: Any

                    ) -> List[Document]:

                        dictionary = self.collection.query(

                            self.embeddings_column, query=query, k=k, lazy=False, **kwargs

                        ).to_dict()

                        documents = dictionary[self.document_column]

                        metadatas = {

                            k: dictionary[k] for k in dictionary if k != self.document_column

                        }

                        out = []

                        for i in range(len(documents)):

                            doc = documents[i]

                            d = {k: metadatas[k][i] for k in metadatas}

                            out.append(Document(page_content=doc, metadata=d))

                        return out

                    @classmethod

                    def from_texts(

                        cls,

                        *args,

                        **kwargs: Any,

                    ):

                        """Return VectorStore initialized from texts and embeddings."""

                        return None

                return VexpressoVectorStore(self)

            @classmethod

            def from_documents(

                cls, documents: List[Document], *args, **kwargs

            ) -> DaftCollection:

                # for langchain integration

                raw = [{"text": d.page_content, **d.metadata} for d in documents]

                return DaftCollection(data=raw, *args, **kwargs)

## Classes

### DaftCollection

```python3
class DaftCollection(
    data: 'Optional[Union[str, pd.DataFrame, Dict[str, Any]]]' = None,
    retriever: 'Retriever' = <vexpresso.retriever.np.NumpyRetriever object at 0x7f2f857d3190>,
    embedding_functions: 'Dict[str, Any]' = {},
    daft_df: 'Optional[daft.DataFrame]' = None
)
```

??? example "View Source"
        class DaftCollection(Collection):

            def __init__(

                self,

                data: Optional[Union[str, pd.DataFrame, Dict[str, Any]]] = None,

                retriever: Retriever = NumpyRetriever(),

                embedding_functions: Dict[str, Any] = {},

                daft_df: Optional[daft.DataFrame] = None,

            ):

                self.daft_df = daft_df

                self.retriever = retriever

                self.embedding_functions = embedding_functions

                _metadata = {}

                if data is not None:

                    if isinstance(data, str):

                        if data.endswith(".json"):

                            with open(data, "r") as f:

                                _metadata = pd.DataFrame(json.load(f))

                    elif isinstance(data, pd.DataFrame):

                        _metadata = data.to_dict("list")

                    else:

                        _metadata = data

                if daft_df is None and len(_metadata) > 0:

                    if isinstance(_metadata, list):

                        self.daft_df = daft.from_pylist(_metadata)

                    else:

                        self.daft_df = daft.from_pydict({**_metadata})

                    self.daft_df = self.daft_df.with_column(

                        "vexpresso_index", indices(col(self.column_names[0]))

                    )

            @property

            def df(self) -> Wrapper:

                return Wrapper(self)

            def __len__(self) -> int:

                return self.daft_df.count_rows()

            def __getitem__(self, column: str) -> DaftCollection:

                return self.select(column)

            def __setitem__(self, column: str, value: List[Any]) -> None:

                self.daft_df = self.add_column(column=value, name=column).df

            def cast(

                self, column: str = None, datatype: DataType = DataType.python()

            ) -> DaftCollection:

                if column is None:

                    columns = [col(c).cast(datatype) for c in self.column_names]

                else:

                    columns = [col(column).cast(datatype)]

                return self.from_daft_df(self.daft_df.select(*columns))

            def add_rows(self, data: List[Dict[str, Any]]) -> DaftCollection:

                dic = self.to_dict()

                for k in dic:

                    for d in data:

                        value = d.get(k, None)

                        dic[k].append(value)

                return self.from_data(dic)

            def set_embedding_function(self, column: str, embedding_function: Transformation):

                self.embedding_functions[column] = embedding_function

            @property

            def column_names(self) -> List[str]:

                return self.daft_df.column_names

            def from_daft_df(self, df: daft.DataFrame) -> DaftCollection:

                return DaftCollection(

                    retriever=self.retriever,

                    embedding_functions=self.embedding_functions,

                    daft_df=df,

                )

            def from_data(self, data: Any) -> DaftCollection:

                return DaftCollection(

                    data=data,

                    retriever=self.retriever,

                    embedding_functions=self.embedding_functions,

                )

            def add_column(self, column: List[Any], name: str = None) -> DaftCollection:

                if name is None:

                    num_columns = len(self.daft_df.column_names)

                    name = f"column_{num_columns}"

                new_df = daft.from_pydict({name: column})

                df = self.daft_df.with_column(name, new_df[name])

                return self.from_daft_df(df)

            def collect(self, in_place: bool = False):

                if in_place:

                    self.daft_df = self.daft_df.collect()

                    return self

                return self.from_daft_df(self.daft_df.collect())

            def execute(self) -> DaftCollection:

                return self.collect()

            def to_pandas(self) -> pd.DataFrame:

                collection = self.execute()

                return collection.daft_df.to_pandas()

            def to_dict(self) -> Dict[str, List[Any]]:

                collection = self.execute()

                return collection.daft_df.to_pydict()

            def to_list(self) -> List[Any]:

                collection = self.execute()

                return list(collection.daft_df.to_pydict().values())

            def show(self, num_rows: int):

                return self.daft_df.show(num_rows)

            @lazy(default=True)

            def sort(self, column, desc=True) -> DaftCollection:

                return self.from_daft_df(self.daft_df.sort(col(column), desc=desc))

            def _embed_queries(

                self,

                queries,

                embedding_function,

                resource_request=ResourceRequest(),

                *args,

                **kwargs,

            ):

                query_embeddings = (

                    daft.from_pydict({"queries": queries})

                    .with_column(

                        "query_embeddings",

                        embedding_function(col("queries"), *args, **kwargs),

                        resource_request=resource_request,

                    )

                    .select("query_embeddings")

                    .collect()

                    .to_pydict()["query_embeddings"]

                )

                return query_embeddings

            @lazy(default=True)

            def query(

                self,

                column: str,

                query: List[Any] = None,

                query_embedding: List[Any] = None,

                filter_conditions: Optional[Dict[str, Dict[str, str]]] = None,

                k: int = None,

                sort: bool = True,

                embedding_fn: Optional[Transformation] = None,

                score_column_name: Optional[str] = None,

                resource_request: ResourceRequest = ResourceRequest(),

                *args,

                **kwargs,

            ) -> Collection:

                if query is not None:

                    query = [query]

                return self.batch_query(

                    column=column,

                    queries=query,

                    query_embeddings=query_embedding,

                    filter_conditions=filter_conditions,

                    k=k,

                    sort=sort,

                    embedding_fn=embedding_fn,

                    score_column_name=score_column_name,

                    resource_request=resource_request,

                    *args,

                    **kwargs,

                )[0]

            @lazy(default=True)

            def batch_query(

                self,

                column: str,

                queries: List[Any] = None,

                query_embeddings: List[Any] = None,

                filter_conditions: Optional[Dict[str, Dict[str, str]]] = None,

                k: int = None,

                sort: bool = True,

                embedding_fn: Optional[Transformation] = None,

                score_column_name: Optional[str] = None,

                resource_request: ResourceRequest = ResourceRequest(),

                *args,

                **kwargs,

            ) -> List[Collection]:

                batch_size = len(queries) if query_embeddings is None else len(query_embeddings)

                if embedding_fn is not None:

                    if column in self.embedding_functions:

                        if embedding_fn != self.embedding_functions[column]:

                            print(

                                "embedding_fn may not be the same as whats in map! Updating what's in map..."

                            )

                    self.embedding_functions[column] = get_embedding_fn(embedding_fn)

                if query_embeddings is None:

                    query_embeddings = self._embed_queries(

                        queries,

                        self.embedding_functions[column],

                        resource_request,

                        *args,

                        **kwargs,

                    )

                dfs = retrieve(

                    batch_size,

                    self.daft_df,

                    column,

                    query_embeddings,

                    self.retriever,

                    k,

                    sort,

                    score_column_name,

                    resource_request,

                )

                for i in range(len(dfs)):

                    if filter_conditions is not None:

                        dfs[i] = FilterHelper.filter(dfs[i], filter_conditions)

                return [self.from_daft_df(df) for df in dfs]

            @lazy(default=True)

            def select(

                self,

                *args,

            ) -> DaftCollection:

                return self.from_daft_df(FilterHelper.select(self.daft_df, *args))

            @lazy(default=True)

            def exclude(

                self,

                *args,

            ) -> DaftCollection:

                return self.from_daft_df(self.daft_df.exclude(*args))

            @lazy(default=True)

            def filter(

                self, filter_conditions: Dict[str, Dict[str, str]], *args, **kwargs

            ) -> DaftCollection:

                return self.from_daft_df(

                    FilterHelper.filter(self.daft_df, filter_conditions, *args, **kwargs)

                )

            @lazy(default=True)

            def apply(

                self,

                transform_fn: Transformation,

                *args,

                to: Optional[str] = None,

                resource_request: ResourceRequest = ResourceRequest(),

                **kwargs,

            ) -> DaftCollection:

                transform_fn = transformation(transform_fn)

                if not isinstance(args[0], DaftCollection):

                    raise TypeError(

                        "first args in apply must be a DaftCollection! use `collection['column_name']`"

                    )

                _args = []

                for _arg in args:

                    if isinstance(_arg, DaftCollection):

                        column = _arg.daft_df.columns[0]

                        _args.append(column)

                    else:

                        _args.append(_arg)

                _kwargs = {}

                for k in kwargs:

                    _kwargs[k] = kwargs[k]

                    if isinstance(_kwargs[k], DaftCollection):

                        # only support first column

                        column = _kwargs[k].daft_df.columns[0]

                        _kwargs[k] = column

                if to is None:

                    to = f"tranformed_{_args[0].name()}"

                df = self.daft_df.with_column(

                    to, transform_fn(*_args, **_kwargs), resource_request=resource_request

                )

                return self.from_daft_df(df)

            @lazy(default=True)

            def embed(

                self,

                column_name: str,

                *args,

                content: Optional[List[Any]] = None,

                embedding_fn: Optional[Transformation] = None,

                update_embedding_fn: bool = True,

                to: Optional[str] = None,

                resource_request: ResourceRequest = ResourceRequest(),

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

                self.embedding_functions[to] = get_embedding_fn(self.embedding_functions[to])

                args = [self[column_name], *args]

                return collection.apply(

                    self.embedding_functions[to],

                    *args,

                    to=to,

                    resource_request=resource_request,

                    **kwargs,

                )

            def save_local(self, directory: str) -> str:

                os.makedirs(directory, exist_ok=True)

                table = self.daft_df.to_arrow()

                pq.write_table(table, os.path.join(directory, "content.parquet"))

            @classmethod

            def from_local_dir(cls, local_dir: str, *args, **kwargs) -> DaftCollection:

                df = daft.read_parquet(os.path.join(local_dir, "content.parquet"))

                return DaftCollection(daft_df=df, *args, **kwargs)

            @classmethod

            def connect(

                cls, address: str = None, cluster_kwargs: Dict[str, Any] = {}, *args, **kwargs

            ) -> DaftCollection:

                if address is None:

                    addy = ray.init(**cluster_kwargs)

                else:

                    addy = ray.init(address=address, **cluster_kwargs)

                daft.context.set_runner_ray(address=addy.address_info["address"])

                return DaftCollection(*args, **kwargs)

            def to_langchain(self, document_column: str, embeddings_column: str):

                from langchain.docstore.document import Document

                from langchain.vectorstores import VectorStore

                class VexpressoVectorStore(VectorStore):

                    def __init__(self, collection: DaftCollection):

                        self.collection = collection

                        self.document_column = document_column

                        self.embeddings_column = embeddings_column

                    def add_texts(

                        self,

                        texts: Iterable[str],

                        metadatas: Optional[List[dict]] = None,

                        **kwargs: Any,

                    ) -> List[str]:

                        if metadatas is None:

                            metadatas = [{} for _ in range(len(texts))]

                        combined = [

                            {self.document_column: t, **m} for t, m in zip(texts, metadatas)

                        ]

                        self.collection = self.collection.add_rows(combined)

                    def similarity_search(

                        self, query: str, k: int = 4, **kwargs: Any

                    ) -> List[Document]:

                        dictionary = self.collection.query(

                            self.embeddings_column, query=query, k=k, lazy=False, **kwargs

                        ).to_dict()

                        documents = dictionary[self.document_column]

                        metadatas = {

                            k: dictionary[k] for k in dictionary if k != self.document_column

                        }

                        out = []

                        for i in range(len(documents)):

                            doc = documents[i]

                            d = {k: metadatas[k][i] for k in metadatas}

                            out.append(Document(page_content=doc, metadata=d))

                        return out

                    @classmethod

                    def from_texts(

                        cls,

                        *args,

                        **kwargs: Any,

                    ):

                        """Return VectorStore initialized from texts and embeddings."""

                        return None

                return VexpressoVectorStore(self)

            @classmethod

            def from_documents(

                cls, documents: List[Document], *args, **kwargs

            ) -> DaftCollection:

                # for langchain integration

                raw = [{"text": d.page_content, **d.metadata} for d in documents]

                return DaftCollection(data=raw, *args, **kwargs)

------

#### Ancestors (in MRO)

* vexpresso.collection.Collection

#### Static methods

    
#### connect

```python3
def connect(
    address: 'str' = None,
    cluster_kwargs: 'Dict[str, Any]' = {},
    *args,
    **kwargs
) -> 'DaftCollection'
```

??? example "View Source"
            @classmethod

            def connect(

                cls, address: str = None, cluster_kwargs: Dict[str, Any] = {}, *args, **kwargs

            ) -> DaftCollection:

                if address is None:

                    addy = ray.init(**cluster_kwargs)

                else:

                    addy = ray.init(address=address, **cluster_kwargs)

                daft.context.set_runner_ray(address=addy.address_info["address"])

                return DaftCollection(*args, **kwargs)

    
#### from_documents

```python3
def from_documents(
    documents: 'List[Document]',
    *args,
    **kwargs
) -> 'DaftCollection'
```

??? example "View Source"
            @classmethod

            def from_documents(

                cls, documents: List[Document], *args, **kwargs

            ) -> DaftCollection:

                # for langchain integration

                raw = [{"text": d.page_content, **d.metadata} for d in documents]

                return DaftCollection(data=raw, *args, **kwargs)

    
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

```python3
df
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

                    num_columns = len(self.daft_df.column_names)

                    name = f"column_{num_columns}"

                new_df = daft.from_pydict({name: column})

                df = self.daft_df.with_column(name, new_df[name])

                return self.from_daft_df(df)

    
#### add_rows

```python3
def add_rows(
    self,
    data: 'List[Dict[str, Any]]'
) -> 'DaftCollection'
```

??? example "View Source"
            def add_rows(self, data: List[Dict[str, Any]]) -> DaftCollection:

                dic = self.to_dict()

                for k in dic:

                    for d in data:

                        value = d.get(k, None)

                        dic[k].append(value)

                return self.from_data(dic)

    
#### apply

```python3
def apply(
    self,
    transform_fn: 'Transformation',
    *args,
    to: 'Optional[str]' = None,
    resource_request: 'ResourceRequest' = ResourceRequest(num_cpus=None, num_gpus=None, memory_bytes=None),
    **kwargs
) -> 'DaftCollection'
```

Apply method, takes in *args and *kwargs columns and applies a transformation function on them. The transformed columns are in format:

transformed_{column_name}

??? example "View Source"
            @lazy(default=True)

            def apply(

                self,

                transform_fn: Transformation,

                *args,

                to: Optional[str] = None,

                resource_request: ResourceRequest = ResourceRequest(),

                **kwargs,

            ) -> DaftCollection:

                transform_fn = transformation(transform_fn)

                if not isinstance(args[0], DaftCollection):

                    raise TypeError(

                        "first args in apply must be a DaftCollection! use `collection['column_name']`"

                    )

                _args = []

                for _arg in args:

                    if isinstance(_arg, DaftCollection):

                        column = _arg.daft_df.columns[0]

                        _args.append(column)

                    else:

                        _args.append(_arg)

                _kwargs = {}

                for k in kwargs:

                    _kwargs[k] = kwargs[k]

                    if isinstance(_kwargs[k], DaftCollection):

                        # only support first column

                        column = _kwargs[k].daft_df.columns[0]

                        _kwargs[k] = column

                if to is None:

                    to = f"tranformed_{_args[0].name()}"

                df = self.daft_df.with_column(

                    to, transform_fn(*_args, **_kwargs), resource_request=resource_request

                )

                return self.from_daft_df(df)

    
#### batch_query

```python3
def batch_query(
    self,
    column: 'str',
    queries: 'List[Any]' = None,
    query_embeddings: 'List[Any]' = None,
    filter_conditions: 'Optional[Dict[str, Dict[str, str]]]' = None,
    k: 'int' = None,
    sort: 'bool' = True,
    embedding_fn: 'Optional[Transformation]' = None,
    score_column_name: 'Optional[str]' = None,
    resource_request: 'ResourceRequest' = ResourceRequest(num_cpus=None, num_gpus=None, memory_bytes=None),
    *args,
    **kwargs
) -> 'List[Collection]'
```

??? example "View Source"
            @lazy(default=True)

            def batch_query(

                self,

                column: str,

                queries: List[Any] = None,

                query_embeddings: List[Any] = None,

                filter_conditions: Optional[Dict[str, Dict[str, str]]] = None,

                k: int = None,

                sort: bool = True,

                embedding_fn: Optional[Transformation] = None,

                score_column_name: Optional[str] = None,

                resource_request: ResourceRequest = ResourceRequest(),

                *args,

                **kwargs,

            ) -> List[Collection]:

                batch_size = len(queries) if query_embeddings is None else len(query_embeddings)

                if embedding_fn is not None:

                    if column in self.embedding_functions:

                        if embedding_fn != self.embedding_functions[column]:

                            print(

                                "embedding_fn may not be the same as whats in map! Updating what's in map..."

                            )

                    self.embedding_functions[column] = get_embedding_fn(embedding_fn)

                if query_embeddings is None:

                    query_embeddings = self._embed_queries(

                        queries,

                        self.embedding_functions[column],

                        resource_request,

                        *args,

                        **kwargs,

                    )

                dfs = retrieve(

                    batch_size,

                    self.daft_df,

                    column,

                    query_embeddings,

                    self.retriever,

                    k,

                    sort,

                    score_column_name,

                    resource_request,

                )

                for i in range(len(dfs)):

                    if filter_conditions is not None:

                        dfs[i] = FilterHelper.filter(dfs[i], filter_conditions)

                return [self.from_daft_df(df) for df in dfs]

    
#### cast

```python3
def cast(
    self,
    column: 'str' = None,
    datatype: 'DataType' = Python
) -> 'DaftCollection'
```

??? example "View Source"
            def cast(

                self, column: str = None, datatype: DataType = DataType.python()

            ) -> DaftCollection:

                if column is None:

                    columns = [col(c).cast(datatype) for c in self.column_names]

                else:

                    columns = [col(column).cast(datatype)]

                return self.from_daft_df(self.daft_df.select(*columns))

    
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

                    self.daft_df = self.daft_df.collect()

                    return self

                return self.from_daft_df(self.daft_df.collect())

    
#### embed

```python3
def embed(
    self,
    column_name: 'str',
    *args,
    content: 'Optional[List[Any]]' = None,
    embedding_fn: 'Optional[Transformation]' = None,
    update_embedding_fn: 'bool' = True,
    to: 'Optional[str]' = None,
    resource_request: 'ResourceRequest' = ResourceRequest(num_cpus=None, num_gpus=None, memory_bytes=None),
    **kwargs
) -> 'DaftCollection'
```

??? example "View Source"
            @lazy(default=True)

            def embed(

                self,

                column_name: str,

                *args,

                content: Optional[List[Any]] = None,

                embedding_fn: Optional[Transformation] = None,

                update_embedding_fn: bool = True,

                to: Optional[str] = None,

                resource_request: ResourceRequest = ResourceRequest(),

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

                self.embedding_functions[to] = get_embedding_fn(self.embedding_functions[to])

                args = [self[column_name], *args]

                return collection.apply(

                    self.embedding_functions[to],

                    *args,

                    to=to,

                    resource_request=resource_request,

                    **kwargs,

                )

    
#### exclude

```python3
def exclude(
    self,
    *args
) -> 'DaftCollection'
```

??? example "View Source"
            @lazy(default=True)

            def exclude(

                self,

                *args,

            ) -> DaftCollection:

                return self.from_daft_df(self.daft_df.exclude(*args))

    
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

                return self.from_daft_df(

                    FilterHelper.filter(self.daft_df, filter_conditions, *args, **kwargs)

                )

    
#### from_daft_df

```python3
def from_daft_df(
    self,
    df: 'daft.DataFrame'
) -> 'DaftCollection'
```

??? example "View Source"
            def from_daft_df(self, df: daft.DataFrame) -> DaftCollection:

                return DaftCollection(

                    retriever=self.retriever,

                    embedding_functions=self.embedding_functions,

                    daft_df=df,

                )

    
#### from_data

```python3
def from_data(
    self,
    data: 'Any'
) -> 'DaftCollection'
```

??? example "View Source"
            def from_data(self, data: Any) -> DaftCollection:

                return DaftCollection(

                    data=data,

                    retriever=self.retriever,

                    embedding_functions=self.embedding_functions,

                )

    
#### query

```python3
def query(
    self,
    column: 'str',
    query: 'List[Any]' = None,
    query_embedding: 'List[Any]' = None,
    filter_conditions: 'Optional[Dict[str, Dict[str, str]]]' = None,
    k: 'int' = None,
    sort: 'bool' = True,
    embedding_fn: 'Optional[Transformation]' = None,
    score_column_name: 'Optional[str]' = None,
    resource_request: 'ResourceRequest' = ResourceRequest(num_cpus=None, num_gpus=None, memory_bytes=None),
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
            @lazy(default=True)

            def query(

                self,

                column: str,

                query: List[Any] = None,

                query_embedding: List[Any] = None,

                filter_conditions: Optional[Dict[str, Dict[str, str]]] = None,

                k: int = None,

                sort: bool = True,

                embedding_fn: Optional[Transformation] = None,

                score_column_name: Optional[str] = None,

                resource_request: ResourceRequest = ResourceRequest(),

                *args,

                **kwargs,

            ) -> Collection:

                if query is not None:

                    query = [query]

                return self.batch_query(

                    column=column,

                    queries=query,

                    query_embeddings=query_embedding,

                    filter_conditions=filter_conditions,

                    k=k,

                    sort=sort,

                    embedding_fn=embedding_fn,

                    score_column_name=score_column_name,

                    resource_request=resource_request,

                    *args,

                    **kwargs,

                )[0]

    
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

                table = self.daft_df.to_arrow()

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

                return self.from_daft_df(FilterHelper.select(self.daft_df, *args))

    
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

                return self.daft_df.show(num_rows)

    
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

                return self.from_daft_df(self.daft_df.sort(col(column), desc=desc))

    
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

                return collection.daft_df.to_pydict()

    
#### to_langchain

```python3
def to_langchain(
    self,
    document_column: 'str',
    embeddings_column: 'str'
)
```

??? example "View Source"
            def to_langchain(self, document_column: str, embeddings_column: str):

                from langchain.docstore.document import Document

                from langchain.vectorstores import VectorStore

                class VexpressoVectorStore(VectorStore):

                    def __init__(self, collection: DaftCollection):

                        self.collection = collection

                        self.document_column = document_column

                        self.embeddings_column = embeddings_column

                    def add_texts(

                        self,

                        texts: Iterable[str],

                        metadatas: Optional[List[dict]] = None,

                        **kwargs: Any,

                    ) -> List[str]:

                        if metadatas is None:

                            metadatas = [{} for _ in range(len(texts))]

                        combined = [

                            {self.document_column: t, **m} for t, m in zip(texts, metadatas)

                        ]

                        self.collection = self.collection.add_rows(combined)

                    def similarity_search(

                        self, query: str, k: int = 4, **kwargs: Any

                    ) -> List[Document]:

                        dictionary = self.collection.query(

                            self.embeddings_column, query=query, k=k, lazy=False, **kwargs

                        ).to_dict()

                        documents = dictionary[self.document_column]

                        metadatas = {

                            k: dictionary[k] for k in dictionary if k != self.document_column

                        }

                        out = []

                        for i in range(len(documents)):

                            doc = documents[i]

                            d = {k: metadatas[k][i] for k in metadatas}

                            out.append(Document(page_content=doc, metadata=d))

                        return out

                    @classmethod

                    def from_texts(

                        cls,

                        *args,

                        **kwargs: Any,

                    ):

                        """Return VectorStore initialized from texts and embeddings."""

                        return None

                return VexpressoVectorStore(self)

    
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

                return list(collection.daft_df.to_pydict().values())

    
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

                return collection.daft_df.to_pandas()