from typing import List

from vexpresso.embedding_functions.base import EmbeddingFunction


class OpenAIEmbeddingFunction(EmbeddingFunction):
    def __init__(self):
        from langchain.embeddings.openai import OpenAIEmbeddings

        self.embeddings_fn = OpenAIEmbeddings()

    def __call__(self, list_of_texts: List[str]):
        """
        This is the main function of `embedding function` to be applied on a column
        """
        return self.embeddings_fn.embed_documents([list_of_texts])
