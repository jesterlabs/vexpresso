import json
import pandas as pd
from langchain.embeddings import HuggingFaceEmbeddings
import numpy as np
from vexpresso.retrieval import (
    RetrievalOutput,
    RetrievalStrategy,
    TopKRetrievalStrategy,
)
from vexpresso.collection import Collection


with open("./data/pokedex.json", "r") as f:
    documents = json.load(f)

df = pd.DataFrame(documents)
descriptions = list(df["description"])
species = list(df["species"])

# combine species + descriptions for better content
content = [f"{s}: {d}" for s, d in zip(species, descriptions)]

names = list(df["name"])
names = [name["english"] for name in names]
df["type"] = df["type"].astype(str) # convert list to string

# using langchain embeddings function
embeddings_fn = HuggingFaceEmbeddings()

collection = Collection(
    content = content,
    ids = names,
    embedding_fn = embeddings_fn,
    metadata = df
)

top_k = collection.query("Loves to sleep", k=10)

for id, content in zip(top_k.ids, top_k.content):
    print(f"{id} -- {content}")


# get df from top_k
df = top_k.metadata.df()

# Filter queried results by pokemon species
print("Filter for psychic type sleepy pokemon")
filter_condition = "type.str.contains('Psychic')"
filtered = top_k.filter(filter_condition)
print(filtered.ids)
