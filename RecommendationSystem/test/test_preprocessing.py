import unittest

from RecommendationSystem.Clustering.Clustering import Clustering
from RecommendationSystem.NewsAgencyScraper.NewsAgencyScraper.items import clean_comment


class TestCleanUpComment(unittest.TestCase):

    def test_clean_up_correct_comments(self):
        comment = "Hi  this is    a test  comment. I am 'writing' this."

        cleaned_up_comment = clean_comment(comment)

        self.assertEqual(cleaned_up_comment, "Hi this is a test comment. I am writing this.")

    def test_clean_up_emtpy_comment(self):
        comment = ""

        cleaned_up_comment = clean_comment(comment)

        self.assertEqual("", cleaned_up_comment)

    def test_preprocess_only_whitespace_comment(self):
        comment = "              "

        cleaned_up_comment = clean_comment(comment)

        self.assertEqual("", cleaned_up_comment)


class TestCommentPreprocessing(unittest.TestCase):
    def test_preprocess_comments_with_different_topics(self):
        comments = ["I am writing test comment 1",
                    "I was reading test comment 2",
                    "strawberry ice cream is the best",
                    "chocolate ice cream is the best"]
        clustering_model = Clustering(None)

        preprocessed_comments = clustering_model.preprocess_comments_for_clustering(comments)

        self.assertEqual([['test', 'comment'],
                          ['test', 'comment'],
                          [ 'ice', 'cream', 'good'],
                          ['chocolate', 'ice', 'cream', 'good']],
                         preprocessed_comments)

    def test_empty_comments_list(self):
        comments = []
        clustering_model = Clustering(None)
        preprocessed_comments = clustering_model.preprocess_comments_for_clustering(comments)

        self.assertEqual([], preprocessed_comments)

    def test_list_with_empty_comments(self):
        comments = ["", "", " "]
        clustering_model = Clustering(None)
        preprocessed_comments = clustering_model.preprocess_comments_for_clustering(comments)

        self.assertEqual([[], [], []], preprocessed_comments)
