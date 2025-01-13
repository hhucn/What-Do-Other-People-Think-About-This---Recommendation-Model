import unittest
from unittest.mock import Mock

from RecommendationSystem.NewsAgencyScraper.NewsAgencyScraper.pipeline_utils import get_existing_topic_or_none
from RecommendationSystem.SemanticSimilarity.SemanticSimilarity import SemanticSimilarity


class TestPipelineUtilities(unittest.TestCase):
    semantic_similarity_model = SemanticSimilarity()

    def test_get_existing_topic_or_none_with_new_topic_expects_none(self):
        topic_1 = Mock(topic_name="politics")
        topic_2 = Mock(topic_name="wind power")
        topic_3 = Mock(topic_name="government")

        keyword = "ice cream"
        all_topics = [topic_1, topic_2, topic_3]

        topic = get_existing_topic_or_none(keyword, None, all_topics, self.semantic_similarity_model)

        self.assertIsNone(topic)

    def test_get_existing_topic_or_none_with_no_new_topic_expecting_topic(self):
        topic_1 = Mock(topic_name="police")
        topic_2 = Mock(topic_name="wind power")
        topic_3 = Mock(topic_name="traveling")

        keyword = "law enforcement"
        all_topics = [topic_1, topic_2, topic_3]

        topic = get_existing_topic_or_none(keyword, None, all_topics, self.semantic_similarity_model)

        self.assertEqual("police", topic.topic_name)



if __name__ == '__main__':
    unittest.main()
