import torch
from sentence_transformers import SentenceTransformer


class EmbeddingModel:

    def __init__(self):

        self.device = (
            "cuda"
            if torch.cuda.is_available()
            else "cpu"
        )

        print(
            f"Embedding device: {self.device}"
        )

        self.model = SentenceTransformer(
            "all-MiniLM-L6-v2",
            device=self.device
        )


    def encode(self, texts):

        return self.model.encode(
            texts,
            normalize_embeddings=True,
            show_progress_bar=True
        )