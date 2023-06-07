# Module vexpresso.main

??? example "View Source"
        from typing import Any, Dict, Optional

        from vexpresso.collection import Collection

        from vexpresso.daft import DaftCollection

        COLLECTION_TYPES = {

            "daft": DaftCollection,

        }

        DEFAULT_COLLECTION = "daft"

        

        def _should_load(

            directory_or_repo_id: Optional[str] = None,

            hf_username: Optional[str] = None,

            repo_name: Optional[str] = None,

        ) -> bool:

            if directory_or_repo_id is None and hf_username is None and repo_name is None:

                return False

            return True

        

        def create(

            collection_type: str = "daft",

            directory_or_repo_id: Optional[str] = None,

            token: Optional[str] = None,

            local_dir: Optional[str] = None,

            to_tmpdir: bool = False,

            hf_username: Optional[str] = None,

            repo_name: Optional[str] = None,

            hub_download_kwargs: Optional[Dict[str, Any]] = {},

            *args,

            **kwargs

        ) -> Collection:

            collection_class = COLLECTION_TYPES.get(

                collection_type, COLLECTION_TYPES[DEFAULT_COLLECTION]

            )

            if _should_load(directory_or_repo_id, hf_username, repo_name):

                return collection_class.load(

                    directory_or_repo_id=directory_or_repo_id,

                    token=token,

                    local_dir=local_dir,

                    to_tmpdir=to_tmpdir,

                    hf_username=hf_username,

                    repo_name=repo_name,

                    hub_download_kwargs=hub_download_kwargs,

                    *args,

                    **kwargs

                )

            return collection_class(*args, **kwargs)

        

        create_collection = create

## Variables

```python3
COLLECTION_TYPES
```

```python3
DEFAULT_COLLECTION
```

## Functions

    
### create

```python3
def create(
    collection_type: str = 'daft',
    directory_or_repo_id: Optional[str] = None,
    token: Optional[str] = None,
    local_dir: Optional[str] = None,
    to_tmpdir: bool = False,
    hf_username: Optional[str] = None,
    repo_name: Optional[str] = None,
    hub_download_kwargs: Optional[Dict[str, Any]] = {},
    *args,
    **kwargs
) -> vexpresso.collection.Collection
```

??? example "View Source"
        def create(

            collection_type: str = "daft",

            directory_or_repo_id: Optional[str] = None,

            token: Optional[str] = None,

            local_dir: Optional[str] = None,

            to_tmpdir: bool = False,

            hf_username: Optional[str] = None,

            repo_name: Optional[str] = None,

            hub_download_kwargs: Optional[Dict[str, Any]] = {},

            *args,

            **kwargs

        ) -> Collection:

            collection_class = COLLECTION_TYPES.get(

                collection_type, COLLECTION_TYPES[DEFAULT_COLLECTION]

            )

            if _should_load(directory_or_repo_id, hf_username, repo_name):

                return collection_class.load(

                    directory_or_repo_id=directory_or_repo_id,

                    token=token,

                    local_dir=local_dir,

                    to_tmpdir=to_tmpdir,

                    hf_username=hf_username,

                    repo_name=repo_name,

                    hub_download_kwargs=hub_download_kwargs,

                    *args,

                    **kwargs

                )

            return collection_class(*args, **kwargs)

    
### create_collection

```python3
def create_collection(
    collection_type: str = 'daft',
    directory_or_repo_id: Optional[str] = None,
    token: Optional[str] = None,
    local_dir: Optional[str] = None,
    to_tmpdir: bool = False,
    hf_username: Optional[str] = None,
    repo_name: Optional[str] = None,
    hub_download_kwargs: Optional[Dict[str, Any]] = {},
    *args,
    **kwargs
) -> vexpresso.collection.Collection
```

??? example "View Source"
        def create(

            collection_type: str = "daft",

            directory_or_repo_id: Optional[str] = None,

            token: Optional[str] = None,

            local_dir: Optional[str] = None,

            to_tmpdir: bool = False,

            hf_username: Optional[str] = None,

            repo_name: Optional[str] = None,

            hub_download_kwargs: Optional[Dict[str, Any]] = {},

            *args,

            **kwargs

        ) -> Collection:

            collection_class = COLLECTION_TYPES.get(

                collection_type, COLLECTION_TYPES[DEFAULT_COLLECTION]

            )

            if _should_load(directory_or_repo_id, hf_username, repo_name):

                return collection_class.load(

                    directory_or_repo_id=directory_or_repo_id,

                    token=token,

                    local_dir=local_dir,

                    to_tmpdir=to_tmpdir,

                    hf_username=hf_username,

                    repo_name=repo_name,

                    hub_download_kwargs=hub_download_kwargs,

                    *args,

                    **kwargs

                )

            return collection_class(*args, **kwargs)