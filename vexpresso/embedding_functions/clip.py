from vexpresso.embedding_functions.base import EmbeddingFunction

DEFAULT_MODEL = "openai/clip-vit-base-patch32"


class ClipEmbeddingsFunction(EmbeddingFunction):
    def __init__(self, model: str = DEFAULT_MODEL):
        import torch
        from transformers import CLIPModel, CLIPProcessor, CLIPTokenizerFast

        self.model = CLIPModel.from_pretrained(model)
        self.processor = CLIPProcessor.from_pretrained(model)
        self.tokenizer = CLIPTokenizerFast.from_pretrained(model)
        self.device = torch.device("cpu")

        if torch.cuda.is_available():
            self.device = torch.device("cuda")
            self.model = self.model.to(self.device)

    def __call__(self, inp, inp_type: str):
        if inp_type == "image":
            inputs = self.processor(images=inp, return_tensors="pt", padding=True)[
                "pixel_values"
            ].to(self.device)
            return self.model.get_image_features(inputs).detach().cpu().numpy()
        if inp_type == "text":
            inputs = self.tokenizer(inp, padding=True, return_tensors="pt")
            inputs["input_ids"] = inputs["input_ids"].to(self.device)
            inputs["attention_mask"] = inputs["attention_mask"].to(self.device)
            return self.model.get_text_features(**inputs).detach().cpu().numpy()
