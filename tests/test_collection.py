from vexpresso.collection import Collection
import numpy as np
from vexpresso.retrieval import TopKRetrievalStrategy

seed = np.random.seed(1337)

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

random_ints = np.random.randint(10, size=(len(texts),))

metadata = {"ints":random_ints}

embeddings = mock_embedding_function(texts)

strategy = TopKRetrievalStrategy("euclidian")

collection = Collection(
    content=texts,
    embeddings=embeddings,
    embedding_fn=mock_embedding_function,
    retrieval_strategy=strategy,
    metadata=metadata
)


def test_query(collection, query = None, query_embedding = None, k=3):
    print("==========================   test_query   ==========================")
    print(f"query: {query}")
    if query_embedding is not None:
        print(f"query_embedding: {query_embedding.shape}")
        query = None
    else:
        print(f"query_embedding: {None}")
    out = collection.query(query=query, query_embedding=query_embedding, k=k)
    print(out.content)
    print("========================== end test_query ==========================")

test_query(collection, ["test1"], None, 3)

test_query(collection, ["test3"], mock_embedding_function(["test3"]), 3)


# print(collection.where("ints", [3,4]))

# query_output = collection.query(query=["test1"], k=3)
# print(query_output.content)

# embedding = mock_embedding_function(["test3"])

# query_output = collection.query(query_embedding=embedding, k=3)
# print(query_output.content)

# query_output = collection.query(query=texts, k=3)
# print(query_output[-1].content)

# query_output = collection.query(query=embeddings, k=3)
# print(query_output[-1].content)

# print("COSINE")

# embeddings = mock_embedding_function(texts)

# strategy = TopKRetrievalStrategy("cosine")

# collection = Collection(
#     content=texts,
#     embeddings=embeddings,
#     embedding_fn=mock_embedding_function,
#     retrieval_strategy=strategy,
# )

# query_output = collection.query(query=["test1"], k=3)
# print("FIRST_ID", query_output.content)

# embedding = mock_embedding_function(["test3"])

# query_output = collection.query(query_embedding=embedding, k=3)
# print(query_output.content)

# query_output = collection.query(query=embeddings, k=3)
# print(query_output[-1].content)

# query_output = collection.query(query=texts, k=3)
# print(query_output[-1].content)
