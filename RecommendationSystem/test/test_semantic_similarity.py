import unittest
from unittest.mock import Mock

from RecommendationSystem.SemanticSimilarity.SemanticSimilarity import SemanticSimilarity


class TestSemanticSimilarity(unittest.TestCase):
    def test_compute_similarity_for_texts_with_similar_texts(self):
        text_1 = "This is a test comment"
        text_2 = "This is another test comment"

        model = SemanticSimilarity()

        similarity = model.compute_similarity_for_texts(text_1, text_2);

        self.assertGreater(similarity, 0.9)

    def test_compute_similarity_for_texts_with_not_similar_texts(self):
        text_1 = "This is a test comment"
        text_2 = "Foo Bar Baz"

        model = SemanticSimilarity()

        similarity = model.compute_similarity_for_texts(text_1, text_2);

        self.assertLess(similarity, 0.2)

    def test_compute_similarity_for_vectors_with_similar_vectors(self):
        vector_1 = ["Foo", "Bar", "Baz"]
        vector_2 = ["Bar", "Baz", "Baz"]

        model = SemanticSimilarity()

        similarity = model.compute_similarity_for_vectors(vector_1, vector_2)

        self.assertGreater(similarity, 0.9)

    def test_compute_similarity_for_vectors_with_no_similar_vectors(self):
        vector_1 = ["Foo", "Bar", "Baz"]
        vector_2 = ["keyword1", "keyword2", "keyword3"]

        model = SemanticSimilarity()

        similarity = model.compute_similarity_for_vectors(vector_1, vector_2)

        self.assertLess(similarity, 0.2)


if __name__ == '__main__':
    unittest.main()
