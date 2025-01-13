import os
from typing import Optional, List

from RecommendationSystem.DB.db_models.topic import Topic


def get_existing_topic_or_none(keyword: str, topic: Optional[Topic], all_topics: List[Topic], semantic_similarity_model) -> \
Optional[Topic]:
    if topic is None:
        for topic in all_topics:
            if semantic_similarity_model.compute_similarity_for_texts(topic.topic_name, keyword) > float(os.getenv(
                    "TOPIC_SIMILARITY_THRESHOLD", 0.75)):
                return topic
        return None
    return topic
