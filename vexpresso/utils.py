import os
from functools import reduce
from typing import Optional


def deep_get(dictionary, keys=None, default=None):
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
            **kwargs
        )

    def upload(
        self,
        repo_id: str,
        folder_path: str,
        token: Optional[str] = None,
        private: bool = True,
        *args,
        **kwargs
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
            **kwargs
        )

    def download(
        self,
        repo_id: str,
        token: Optional[str] = None,
        local_dir: Optional[str] = None,
        *args,
        **kwargs
    ) -> str:
        if token is None:
            token = os.environ.get("HUGGINGFACEHUB_API_TOKEN")
        return self._hf_hub.snapshot_download(
            repo_id=repo_id,
            token=token,
            local_dir=local_dir,
            repo_type="dataset",
            *args,
            **kwargs
        )
