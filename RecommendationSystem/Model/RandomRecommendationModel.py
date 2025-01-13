import os
import random
from typing import List

from dotenv import load_dotenv

from RecommendationSystem.DB.utils import get_all_comments_by_article_id

load_dotenv()
random.seed(os.getenv("SEED"))


class RandomRecommendationModel:
    def get_recommendations(self, article_id: int) -> List:
        comments = get_all_comments_by_article_id(article_id)
        random.shuffle(comments)

        return [comment.text for comment in comments][:3]
