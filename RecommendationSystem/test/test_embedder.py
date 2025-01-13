import os
import unittest
from datetime import datetime
from unittest import mock

import torch

from RecommendationSystem.DB.db_models.article import Article
from RecommendationSystem.DB.db_models.cluster import Cluster
from RecommendationSystem.DB.db_models.comment import Comment
from RecommendationSystem.Embedder.embedding_model import EmbeddingModel
from RecommendationSystem.Embedder.run_embedder import main


class TestEmbedder(unittest.TestCase):
    @mock.patch.dict(os.environ, {"SEMANTIC_SIMILARITY_MODEL": "all-mpnet-base-v2"})
    def test_embed_texts_with_text_should_succeed(self):
        text = "I am a text"
        embedder = EmbeddingModel()

        embedded_text = embedder.embed_texts(text)

        self.assertAlmostEqual(embedded_text[0].item(), 0.0353, 3)

    @mock.patch.dict(os.environ, {"SEMANTIC_SIMILARITY_MODEL": "all-mpnet-base-v2"})
    def test_embed_vector_with_vector_should_succeed(self):
        vector = ["keyword1", "foo", "bar", "baz"]
        embedder = EmbeddingModel()

        embedded_vector = embedder.embed_vector(vector)

        self.assertAlmostEqual(embedded_vector[0].item(), 0.0013, places=3)


if __name__ == '__main__':
    unittest.main()
