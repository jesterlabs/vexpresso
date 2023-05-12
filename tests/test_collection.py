from vexpresso.collection import Collection
import numpy as np
from vexpresso.query import NumpyQueryStrategy

def mock_embedding_function(texts):
    o = []
    for text in texts:
        if "test" in text:
            split_val = float(text.split("test")[-1])
            embedding = np.ones((768,))*split_val
        else:
            embedding = np.zeros((768,))
        o.append(embedding)
    return np.stack(o)

texts = [f"test{i}" for i in range(100)]

print("EUCLIDIAN")

embeddings = mock_embedding_function(texts)

strategy = NumpyQueryStrategy("euclidian")

collection = Collection(
    content=texts,
    embeddings=embeddings,
    embedding_fn=mock_embedding_function,
    query_strategy=strategy
)

query_output = collection.query(query=["test1"], k=3)
print(query_output.ids)

embedding = mock_embedding_function(["test3"])

query_output = collection.query(query_embedding=embedding, k=3)
print(query_output.ids)

query_output = collection.query(query=texts, k=3)
print(query_output[-1].ids)

query_output = collection.query(query=embeddings, k=3)
print(query_output[-1].ids)

print("COSINE")

embeddings = mock_embedding_function(texts)

strategy = NumpyQueryStrategy("cosine")

collection = Collection(
    content=texts,
    embeddings=embeddings,
    embedding_fn=mock_embedding_function,
    query_strategy=strategy
)

query_output = collection.query(query=["test1"], k=3)
print(query_output.ids)

embedding = mock_embedding_function(["test3"])

query_output = collection.query(query_embedding=embedding, k=3)
print(query_output.ids)

query_output = collection.query(query=embeddings, k=3)
print(query_output[-1].ids)

query_output = collection.query(query=texts, k=3)
print(query_output[-1].ids)