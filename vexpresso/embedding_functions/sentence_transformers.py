from typing import List

from vexpresso.embedding_functions.base import EmbeddingFunction

DEFAULT_MODEL = "sentence-transformers/all-mpnet-base-v2"


class SentenceTransformerEmbeddingFunction(EmbeddingFunction):
    def __init__(
        self, model: str = DEFAULT_MODEL, output_type: str = "np", *args, **kwargs
    ):
        from sentence_transformers import SentenceTransformer

        self.model = SentenceTransformer(model, *args, **kwargs)
        self.output_type = output_type

    def __call__(self, list_of_texts: List[str]):
        out = self.model.encode(list_of_texts, convert_to_tensor=True)
        if self.output_type == "np":
            return out.detach().cpu().numpy()
        if self.output_type == "list":
            return out.tolist()
        return out
