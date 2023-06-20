import gradio as gr
import json
import pandas as pd
from PIL import Image
import requests
import numpy as np
import daft
import vexpresso
from vexpresso.utils import ResourceRequest
from PIL import Image
import requests
from transformers import CLIPModel, CLIPProcessor, CLIPTokenizerFast
import torch
    
class ClipEmbeddingsFunction:
    def __init__(self):

        self.model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
        self.processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
        self.tokenizer = CLIPTokenizerFast.from_pretrained("openai/clip-vit-base-patch32")
        self.device = torch.device('cpu')

        if torch.cuda.is_available():
            self.device = torch.device('cuda')
            self.model = self.model.to(self.device)

    def __call__(self, inp, inp_type):
        if inp_type == "image":
            inputs = self.processor(images=inp, return_tensors="pt", padding=True)['pixel_values'].to(self.device)
            return self.model.get_image_features(inputs).detach().cpu().numpy()
        if inp_type == "text":
            inputs = self.tokenizer(inp, padding=True, return_tensors="pt")
            inputs["input_ids"] = inputs["input_ids"].to(self.device)
            inputs["attention_mask"] = inputs["attention_mask"].to(self.device)
            return self.model.get_text_features(**inputs).detach().cpu().numpy()

def download_images(images, image_type):
    return [Image.open(requests.get(im["hires"], stream=True).raw).convert(image_type) for im in images]

def add_filter(state, filter_var, filter_method, filter_value):
    f = {filter_var:{filter_method:filter_value}}
    state.append(f)
    return f"{state}"

def remove_row(state, row):
    row = max(row, 0)
    if int(row) < len(state) and len(state) > 0:
        del state[int(row)]
    return f"{state}"


if __name__ == "__main__":
    
    with open("./data/pokedex.json", 'r') as f:
        stuff = json.load(f)

    df = pd.DataFrame(stuff)

    # include only Kanto
    df = df.iloc[:151]

    # create collection
    collection = vexpresso.create(data=df, backend="ray")
    resource_request = ResourceRequest(num_gpus=0) # change this to 1 to use gpus

    # download the images
    print("Downoading images and embedding images...")
    collection = collection.apply(
            download_images,
            collection["image"],
            image_type="RGB",
            to="downloaded_image",
            datatype=daft.DataType.python()
        ).embed(
            "downloaded_image",
            embedding_fn=ClipEmbeddingsFunction,
            inp_type="image",
            to="clip_embeddings",
            resource_request = resource_request
    ).execute()

    def find_image_vectors(text_query, image_query, state):
        if text_query is None and image_query is None:
            raise ValueError("Image or text query must be provided")
        embeddings = []
        if text_query is not None:
            embeddings.append(collection.embed_query(text_query, embedding_fn="clip_embeddings", inp_type="text"))
        if image_query is not None:
            embeddings.append(collection.embed_query(image_query, embedding_fn="clip_embeddings", inp_type="image"))

        if len(embeddings) > 1:
            query_embedding = 0.5*embeddings[0] + 0.5*embeddings[1]
        else:
            query_embedding = embeddings[0]

        queried = collection.query(
            "clip_embeddings",
            query_embedding=query_embedding,
            k=10,
            inp_type="text",
        ).execute()
        if state is not None and len(state) > 0:
            for filt in state:
                queried = queried.filter(filter_conditions=filt)
        images = queried["downloaded_image"].to_list()[:4]
        return images


    with gr.Blocks() as demo:
        state = gr.State([])
        with gr.Row():
            with gr.Column():
                gr.Markdown(
                """
                    ### Query Input! Add either a text prompt or upload an image, or add both to average the predictions.
                """)
                vector_query = gr.Textbox(placeholder="Type in a text prompt! Ex: Looks like a plant", show_label=False)
                image_query = gr.Image(show_label=False)
            with gr.Column():
                gr.Markdown(
                """
                    ### Filter method. Use this to filter based on metadata fields
                """)
                filter_var = gr.Textbox(label="filter_var")
                filter_method = gr.Dropdown(choices=[
                        "eq", "neq", "gt", "gte", "lt", "lte", "isin", "notin", "contains", "notcontains"
                    ],
                    label="filter_method"
                )
                filter_value = gr.Textbox(label="filter_value")
            with gr.Column():
                gr.Markdown(
                """
                    ### Current Filter Methods
                """)
                current_filters = gr.Textbox(label="Current Filters")
            filter_button = gr.Button("Add filter")
            filter_button.click(fn=add_filter, inputs=[state, filter_var, filter_method, filter_value], outputs=current_filters)

        with gr.Row():
            button = gr.Button("Submit")
        with gr.Row():
            gallery = gr.Gallery(
                label="Queried Pokemon!", show_label=False, elem_id="gallery", preview=True
            ).style(columns=[2], rows=[2], object_fit="contain", height="auto")
        button.click(find_image_vectors, inputs=[vector_query, image_query, state], outputs=[gallery])
        gr.Examples(
            examples=[
                ["Turtle pokemon, has blue skin", None, []],
                [None, "data/gradio-demo/mewtwo.jpeg", []],
                ["Looks like a plant", "data/gradio-demo/bulbasaur.png", []],
                [None, "data/gradio-demo/pikachu-dog.jpg", []],
                [None, "data/gradio-demo/charmander.png", []],
            ],
            inputs=[vector_query, image_query, state],
            # outputs=[gallery],
            # fn=find_image_vectors,
            cache_examples=False,
        )

    demo.launch()