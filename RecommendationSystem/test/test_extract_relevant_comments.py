import os
import unittest
from multiprocessing import Pool, cpu_count

import numpy as np
from neomodel import Database
from sentence_transformers import SentenceTransformer

from RecommendationSystem.DB.db_models.StanceDetection.ProComment import ProComment
from RecommendationSystem.DB.db_models.comment import Comment
from RecommendationSystem.DB.utils import __extract_results
from RecommendationSystem.Model.RecommendationModel import RecommendationModel


class TestComment:

    def __init__(self, embedding, relevance_score, text):
        self.embedding = embedding
        self.relevance_score = relevance_score
        self.text = text

class MyTestCase(unittest.TestCase):

    def setUp(self):
        self.embedding_model = SentenceTransformer("all-mpnet-base-v2")
        self.comment_1 = Comment(
            text="I am the first comment",
            embedding = self.embedding_model.encode("I am the first comment", convert_to_tensor=True),
            relevance_score=0.5,
        )

        self.comment_2 = Comment(
            text="I am the second comment",
            embedding=self.embedding_model.encode("I am the second comment", convert_to_tensor=True),
            relevance_score=0.1,
        )

        self.comment_3 = Comment(
            text="I am the third comment",
            embedding=self.embedding_model.encode("I am the third comment", convert_to_tensor=True),
            relevance_score=0.8,
        )


    def test_something(self):
        embedding_model = SentenceTransformer("all-mpnet-base-v2")
        os.environ["TOKENIZERS_PARALLELISM"] = "true"
        user_comment_embedding = embedding_model.encode("foo bar baz", convert_to_tensor=True)

        comments_from_cluster = [self.comment_1, self.comment_2, self.comment_3]

        data = [(user_comment_embedding, comments_from_cluster), (user_comment_embedding, comments_from_cluster)]
        foo_1 = None
        with Pool(processes=cpu_count()) as pool:
            #extracted_comments = pool.starmap(RecommendationModel.extract_relevant_comments, data)
            foo_1 = pool.apply(RecommendationModel.extract_relevant_comments, args=(user_comment_embedding, comments_from_cluster,))
            foo_2 = pool.apply(RecommendationModel.extract_relevant_comments, args=(user_comment_embedding, comments_from_cluster,))


        print(foo_1)
        print(foo_2)
        self.assertEqual(True, True)  # add assertion here


if __name__ == '__main__':
    unittest.main()
