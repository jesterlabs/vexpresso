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
    *args,
    collection_type: str = "daft",
    directory_or_repo_id: Optional[str] = None,
    token: Optional[str] = None,
    local_dir: Optional[str] = None,
    to_tmpdir: bool = False,
    hf_username: Optional[str] = None,
    repo_name: Optional[str] = None,
    hub_download_kwargs: Optional[Dict[str, Any]] = {},
    backend: str = "python",
    cluster_address: Optional[str] = None,
    cluster_kwargs: Dict[str, Any] = {},
    **kwargs,
) -> Collection:
    BACKEND_SET = {"python", "ray"}

    collection_class = COLLECTION_TYPES.get(
        collection_type, COLLECTION_TYPES[DEFAULT_COLLECTION]
    )
    if backend not in BACKEND_SET:
        backend = "python"

    if backend == "ray":
        _ = collection_class.connect(cluster_address, cluster_kwargs)

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
            **kwargs,
        )
    collection = collection_class(*args, **kwargs)
    return collection


def connect(
    collection_type: str = "daft",
    cluster_address: Optional[str] = None,
    cluster_kwargs: Dict[str, Any] = {},
    *args,
    **kwargs,
) -> Collection:
    return create(
        collection_type=collection_type,
        cluster_address=cluster_address,
        cluster_kwargs=cluster_kwargs,
        *args,
        **kwargs,
    )


create_collection = create
