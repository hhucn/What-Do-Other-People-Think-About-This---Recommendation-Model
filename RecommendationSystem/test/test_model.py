import os
import sys
import unittest
from unittest import mock, skip
from unittest.mock import Mock

import numpy as np
import torch
from neomodel import config

from RecommendationSystem.DB.db_models.article import Article
from RecommendationSystem.DB.db_models.cluster import Cluster
from RecommendationSystem.DB.db_models.comment import Comment
from RecommendationSystem.DB.db_models.topic import Topic
from RecommendationSystem.Embedder.embedding_model import EmbeddingModel
from RecommendationSystem.Model.StanceRecommendationModel import StanceRecommendationModel
from RecommendationSystem.SemanticSimilarity.SemanticSimilarity import SemanticSimilarity

ROOT_DIR = os.path.abspath(os.path.join(
    os.path.dirname(__file__),
    os.path.pardir
))

sys.path.append(ROOT_DIR + "/API")
config.DATABASE_URL = os.environ.get("NEO4J_BOLT_URL")

from RecommendationSystem.Model.RecommendationModel import RecommendationModel


class TestModel(unittest.TestCase):

    def setUp(self) -> None:
        self.topic_1 = Topic(topic_name="law enforcement")
        self.topic_2 = Topic(topic_name="police officer")
        self.topic_3 = Topic(topic_name="wind power plants")
        self.topic_4 = Topic(topic_name="sport")

        self.article_1: Article = Article(
            article_title='Article 1',
            news_agency='Foo Bar News',
            keywords='Keyword1 Keyword2',
            embedding=[1, 1, 1]
        )

        self.article_2: Article = Article(article_title='Article 2', news_agency='Foo Bar News',
                                          keywords='Keyword3 Keyword4',
                                          embedding=[1, 1, 1]
                                          )

        self.article_3: Article = Article(
            article_title='Article 3',
            news_agency='Foo Bar News',
            keywords='Keyword5 Keyword6',
            embedding=[1, 1, 1]
        )

        self.cluster_1: Cluster = Cluster(cluster_name="cluster 1", most_frequent_words=["Foo", "Bar", "Baz"], embedding=np.array([1., 1., 1.]).astype('float32'))
        self.cluster_2: Cluster = Cluster(cluster_name="cluster 2", most_frequent_words=["Baz", "Baz"], embedding=np.array([1., 0., 1.]).astype('float32'))
        self.cluster_3: Cluster = Cluster(cluster_name="cluster 3", most_frequent_words=["test 1", "test 2"], embedding=np.array([0., 1., 0.]).astype('float32'))

        self.comment_1: Comment = Comment(text="I am test comment 1", relevance_score=0.8, embedding=[1., 0.8])
        self.comment_2: Comment = Comment(text="I am test comment 2", relevance_score=1.3, embedding=[1., 0.9])
        self.comment_3: Comment = Comment(text="I am test comment 3", relevance_score=0.1, embedding=[0., 0.])
        self.comment_4: Comment = Comment(text="I am test comment 4", relevance_score=1.8, embedding=[1., 1.])

    def test_find_most_suitable_article(self):
        keywords = [torch.tensor([1., 1., 1.]), torch.tensor([1., 1., 1.]), torch.tensor([0., 0., 0.])]
        all_article = [self.article_1, self.article_2, self.article_3]
        model = StanceRecommendationModel(None, None)
        model.number_of_article = 2

        articles = model.find_most_suitable_article(keywords, all_article)

        given_article_title = [article.article_title for article in articles]
        expected_article_title = ["Article 1", "Article 2"]

        self.assertListEqual(given_article_title, expected_article_title)

    def test_get_cluster_for_keywords(self):
        user_comment = "This is a test comment"
        all_clusters = np.array([self.cluster_1, self.cluster_2, self.cluster_3])

        embedding_model = Mock(EmbeddingModel)
        similarity_model = Mock(SemanticSimilarity)
        model = StanceRecommendationModel(similarity_model, embedding_model)

        embedding_model.embed_vector.return_value = torch.tensor([1., 1., 1.])

        clusters = model.get_cluster_for_user_comment(user_comment, all_clusters, threshold=0.6)

        given_cluster_names = [cluster.cluster_name for cluster in clusters]
        expected_cluster_names = ["cluster 1", "cluster 2"]

        self.assertListEqual(given_cluster_names, expected_cluster_names)




if __name__ == '__main__':
    unittest.main()
