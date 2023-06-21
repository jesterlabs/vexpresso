<div class="header" align="center"><h1 style="display: inline;"><img src="https://github.com/shyamsn97/vexpresso/blob/main/docs/_static/cup.png" alt="drawing" width="50" height="50" align="center">Vexpresso</h1><p>Vexpresso is a simple and scalable multi-modal vector database built with <a href="https://www.getdaft.io/">Daft</a></p></div>

<figure>
<!-- <video width="320" height="320" controls>
  <source src="docs/_static/PokemonGradio.mp4" type="video/mp4">
</video> -->
<img src="https://github.com/shyamsn97/vexpresso/blob/main/docs/_static/PokemonGradio.gif">
<figcaption>Querying Pokemon with images and text</figcaption>
</figure>

## Features
🍵  **Simple**: Vexpresso is lightweight and is very easy to get started!

🔌  **Flexible**: Unlike many other vector databases, Vexpresso supports arbitrary datatypes. This means that you can query muti-modal objects (images, audio, video, etc...)

🌐 **Scalable**: Because Vexpresso uses [Daft](https://www.getdaft.io/), it can be scaled using [Ray](https://www.ray.io/) to multi-gpu / cpu clusters.

📚 **Persistent**: Easy Saving and Loading functionality: Vexpresso has easily accessible functions for saving / loading to huggingface datasets.

## Installation
To install from PyPi:

```pip install vexpresso```

To install from source:

```
git clone git@github.com:shyamsn97/vexpresso.git
cd vexpresso
pip install -e .
```

## Usage

> 🔥 Check out our [Showcase](./examples/Showcase.ipynb) notebook for a more detailed walkthrough!

In this simple example, we create a simple collection and embed using huggingface sentence transformers.

```python
from typing import List, Any
import vexpresso
# import embedding functions from vexpresso
import vexpresso.embedding_functions as ef

# creating a collection object!
collection = vexpresso.create(
    data = {
        "documents":[
            "This is document1",
            "This is document2",
            "This is document3",
            "This is document4",
            "This is document5",
            "This is document6"
        ],
        "source":["notion", "google-docs", "google-docs", "notion", "google-docs", "google-docs"],
        "num_lines":[10, 20, 30, 40, 50, 60]
    }
    # backend="ray" # turn this flag on to start / connect to a ray cluster!
)

# create a simple embedding function from sentence_transformers
def hf_embed_fn(content: List[Any]):
    from sentence_transformers import SentenceTransformer
    model = SentenceTransformer("sentence-transformers/all-mpnet-base-v2")
    return model.encode(content, convert_to_tensor=True).detach().cpu().numpy()

# or use a langchain embedding function
def langchain_embed_fn(content: List[Any]):
    from langchain.embeddings import OpenAIEmbeddings
    embeddings_model = OpenAIEmbeddings()
    return embeddings_model.embed_documents(content)

# embed function creates a column in the collection with embeddings. There can be more than one embedding column!
# lazy execution until .execute is called
collection = collection.embed(
    "documents",
    embedding_fn=hf_embed_fn,
    to="document_embeddings",
    # lazy=False # if this is false, execute doesn't need to be called
).execute()

# creating a queried collection with a subset of content closest to query
queried_collection = collection.query(
    "document_embeddings",
    query="query document6",
    k = 4, # return 2 closest
    lazy=False
    # query_embedding=[query1, query2, ...]
    # filter_conditions={"metadata_field":{"operator, ex: 'eq'":"value"}} # optional metadata filter
)

# batch query -- return a list of collections
# batch_queried_collection = collection.batch_query(
#     "document_embeddings",
#     queries=["doc1", "doc2"],
#     k = 2
# )

# filter collection for documents with num_lines less than or equal to 30
filtered_collection = queried_collection.filter(
    {
        "num_lines": {"lte":30}
    }
).execute()

# show dataframe
filtered_collection.show()

# convert to dictionary
filtered_dict = filtered_collection.to_dict()
documents = filtered_dict["documents"]

# add an entry!
collection = collection.add(
    [
        {"documents":"new documents 1", "source":"notion", "num_lines":2},
        {"documents":"new documents 2", "source":"google-docs", "num_lines":40}
    ]
)
collection = collection.execute()
```

## Resources
- [Daft](https://www.getdaft.io/)
- [Ray](https://www.ray.io/)
- [Vector Database Intro](https://www.pinecone.io/learn/vector-database/)

## Contributing

Feel free to make a pull request or open an issue for a feature!



