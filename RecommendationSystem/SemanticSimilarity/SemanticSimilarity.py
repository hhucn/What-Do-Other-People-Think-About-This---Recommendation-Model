from typing import List

import numpy as np
import torch
from sentence_transformers import SentenceTransformer, util


class SemanticSimilarity:
    def __init__(self):
        self.model = SentenceTransformer("all-mpnet-base-v2")

    def compute_similarity_for_texts(self, text_1: str, text_2: str):
        embedding_text_1 = self.model.encode([text_1], convert_to_tensor=True)
        embedding_text_2 = self.model.encode([text_2], convert_to_tensor=True)

        return util.cos_sim(embedding_text_1, embedding_text_2).item()

    def compute_similarity_for_vectors(self, vector_1: List[str], vector_2: List[str]) -> float:
        embedding_vector_1 = torch.mean(self.model.encode(vector_1, convert_to_tensor=True), dim=0)
        embedding_vector_2 = torch.mean(self.model.encode(vector_2, convert_to_tensor=True), dim=0)

        return util.cos_sim(embedding_vector_1, embedding_vector_2)

    @staticmethod
    def compute_similarity(embedding_1: torch.Tensor, embedding_2: torch.Tensor):
        return util.cos_sim(embedding_1, embedding_2)

