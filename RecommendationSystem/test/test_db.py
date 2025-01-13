from datetime import datetime
import os
import unittest
from typing import List

import torch
from neomodel import config

from RecommendationSystem.DB.db_models.article import Article
from RecommendationSystem.DB.db_models.cluster import Cluster
from RecommendationSystem.DB.db_models.comment import Comment
from RecommendationSystem.DB.utils import get_article_by_title, get_all_article, get_article_by_id, \
    get_comment_by_id, get_all_comments_for_given_article, run_cypher_query, get_articles_without_embedding, \
    get_comments_without_embedding, update_node_with_embedding

config.DATABASE_URL = os.environ.get("NEO4J_BOLT_URL")


class TestDBUtils(unittest.TestCase):
    def setUp(self) -> None:
        """
        Prepares neo4j db with entries
        :return:
        """

        self.article_1 = Article(
            article_title="Article 1",
            news_agency="FooBarNews",
            keywords=["Keyword1", "Keyword2"],
            embedding=[1, 2, 3],
            pub_date=datetime(year=2022, month=1, day=1),
        ).save()

        self.article_2 = Article(
            article_title="Article 2",
            news_agency="FooBarNews",
            keywords=["Keyword2", "Keyword3"],
            pub_date=datetime(year=2022, month=1, day=1),
        ).save()

        self.article_3 = Article(
            article_title="Article 3",
            news_agency="FooBarNews",
            keywords=["Keyword4", "Keyword9"],
            pub_date=datetime(year=2022, month=1, day=1),
        ).save()

        self.comment_1 = Comment(
            text="I am a comment 1",
            embedding=[1, 2, 3]
        ).save()

        self.comment_2 = Comment(
            text="I am a comment 2"
        ).save()

        self.comment_3 = Comment(
            text="I am a comment 3",
            embedding=[1, 2, 3]
        ).save()

        self.comment_4 = Comment(
            text="I am a comment 4",
            embedding=[1, 2, 3]
        ).save()

        self.comment_5 = Comment(
            text="I am a comment 5",
            embedding=[1, 2, 3]
        ).save()

        self.cluster_1 = Cluster(
           cluster_name = "cluster 1",
           most_frequent_words = ["foo", "bar", "baz"]
        ).save()
        self.cluster_2 = Cluster(
            cluster_name="cluster 2",
            most_frequent_words=["foo", "bar", "baz"]
        ).save()
        self.cluster_3 = Cluster(
            cluster_name="cluster 3",
            most_frequent_words=["foo", "bar", "baz"]
        ).save()

        self.article_1.cluster.connect(self.cluster_1)
        self.article_2.cluster.connect(self.cluster_1)

        self.cluster_1.comment.connect(self.comment_1)
        self.cluster_1.comment.connect(self.comment_2)
        self.cluster_1.comment.connect(self.comment_4)
        self.cluster_2.comment.connect(self.comment_3)
        self.cluster_3.comment.connect(self.comment_5)

    def tearDown(self) -> None:
        """
        Cleans up database after every test
        :return:
        """
        for article in Article.nodes.all():
            article.cluster.disconnect_all()
            article.delete()

        for cluster in Cluster.nodes.all():
            cluster.article.disconnect_all()
            cluster.comment.disconnect_all()
            cluster.delete()

        for comment in Comment.nodes.all():
            comment.delete()

    def test_get_article_by_title_with_embedding(self):
        article: Article = get_article_by_title("Article 1")
        self.assertEqual(article.article_title, "Article 1")

    def test_get_article_by_title_without_embedding(self):
        article: Article = get_article_by_title("Article 2")
        self.assertIsNone(article)

    def test_get_article_by_id_with_embedding(self):
        article: Article = get_article_by_id(self.article_1.article_id)
        self.assertEqual(article.article_title, self.article_1.article_title)

    def test_get_article_by_id_without_embedding(self):
        article: Article = get_article_by_id(self.article_2.article_id)
        self.assertIsNone(article)

    def test_get_all_article(self):
        articles: List[Article] = get_all_article()
        self.assertEqual(articles[0].article_id, self.article_1.article_id)

    def test_get_comment_by_id_with_embedding(self):
        comment: Comment = get_comment_by_id(self.comment_1.comment_id)
        self.assertEqual(comment.text, self.comment_1.text)

    def test_get_comment_by_id_without_embedding(self):
        comment: Comment = get_comment_by_id(self.comment_2.comment_id)
        self.assertIsNone(comment)

    def test_get_all_comments_for_given_article(self):
        comments: List[Comment] = get_all_comments_for_given_article(self.article_1)
        self.assertIn(self.comment_1, comments)
        self.assertNotIn(self.comment_2, comments)
        self.assertNotIn(self.comment_3, comments)

    def test_run_cypher_query(self):
        comments = run_cypher_query(
            f"""
            MATCH (a:Article)<--(k:Cluster)<--(c:Comment)
            WHERE a.article_id='{self.article_1.article_id}'
            RETURN c
            """,
            Comment
        )
        self.assertIn(self.comment_1, comments)
        self.assertIn(self.comment_2, comments)
        self.assertNotIn(self.comment_3, comments)

    def test_get_article_without_embedding(self):
        articles: List[Article] = get_articles_without_embedding()
        self.assertNotIn(self.article_1, articles)
        self.assertIn(self.article_2, articles)
        self.assertIn(self.article_3, articles)

    def test_get_all_comments_without_embedding(self):
        comments: List[Comment] = get_comments_without_embedding()
        self.assertNotIn(self.comment_1, comments)
        self.assertIn(self.comment_2, comments)
        self.assertNotIn(self.comment_3, comments)

    def test_update_article_node_with_embedding(self):
        self.assertIsNone(self.article_2.embedding)
        embedding = torch.Tensor([1, 2, 3])
        update_node_with_embedding(self.article_2, embedding)
        self.assertIsNotNone(self.article_2.embedding)


if __name__ == '__main__':
    unittest.main()
