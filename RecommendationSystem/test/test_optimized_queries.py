import os
import unittest

from neomodel import config

from RecommendationSystem.DB.utils import get_stance_comments_for_given_clusters, get_sentiment_comments_for_given_clusters, get_emotion_comments_for_given_clusters, get_news_agency_comments_for_given_cluster

config.DATABASE_URL = os.environ.get("NEO4J_BOLT_URL")


class TestCluster:
    def __init__(self, uid):
        self.uid = uid


class MyTestCase(unittest.TestCase):
    def test_something(self):
        test_cluster = [TestCluster(uid="a456769a3ab24a2e8460f0f0474cf744"), TestCluster(uid="a68ae1b76cf143f3bfee61855c00489b")]

        pro_comments, contra_comments = get_stance_comments_for_given_clusters(test_cluster)

        self.assertEqual(True, False)  # add assertion here

    def test_sentiment_queries(self):
        test_cluster = [TestCluster(uid="a456769a3ab24a2e8460f0f0474cf744"), TestCluster(uid="a68ae1b76cf143f3bfee61855c00489b")]

        positive, neutral, negative = get_sentiment_comments_for_given_clusters(test_cluster)

        self.assertEqual(True, False)  # add assertion here

    def test_emotion_queries(self):
        test_cluster = [TestCluster(uid="a456769a3ab24a2e8460f0f0474cf744"), TestCluster(uid="a68ae1b76cf143f3bfee61855c00489b")]

        love, fear, anger, joy, surprise, sadness = get_emotion_comments_for_given_clusters(test_cluster)

        self.assertEqual(True, False)  # add assertion here

    def test_ny_times_queries(self):
        test_cluster = [TestCluster(uid="a456769a3ab24a2e8460f0f0474cf744"), TestCluster(uid="a68ae1b76cf143f3bfee61855c00489b")]

        times, breitbart = get_news_agency_comments_for_given_cluster(test_cluster, test_cluster)
        self.assertEqual(True, False)  # add assertion here


if __name__ == '__main__':
    unittest.main()
