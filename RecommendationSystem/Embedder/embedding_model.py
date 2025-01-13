import os
from typing import List

import torch

from sentence_transformers import SentenceTransformer, util


class EmbeddingModel:
    def __init__(self):
        self.model = SentenceTransformer("all-mpnet-base-v2")

    def embed_texts(self, text: str) -> torch.Tensor:
        return self.model.encode(text, convert_to_tensor=True)

    def embed_vector(self, vector: List[str]) -> torch.Tensor:
        return torch.mean(self.model.encode(vector, convert_to_tensor=True), dim=0)



