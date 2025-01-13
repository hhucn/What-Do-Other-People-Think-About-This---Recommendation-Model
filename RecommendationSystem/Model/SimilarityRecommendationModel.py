import os
import random
from typing import List

from dotenv import load_dotenv

from RecommendationSystem.DB.utils import get_all_comments_by_article_id
from RecommendationSystem.Embedder.embedding_model import EmbeddingModel
from RecommendationSystem.SemanticSimilarity.SemanticSimilarity import SemanticSimilarity

load_dotenv()
random.seed(os.getenv("SEED"))


class SimilarityRecommendationModel:
    def __init__(self, embedding_model: EmbeddingModel, similarity_model: SemanticSimilarity):
        self.embedding_model: EmbeddingModel = embedding_model
        self.similarity_model: SemanticSimilarity = similarity_model

    def get_recommendations(self, article_id: int, user_comment: str) -> List:
        print("Getting similarity recommendations for article id {}".format(article_id))
        comments = get_all_comments_by_article_id(article_id)
        user_comment_embedding = self.embedding_model.embed_texts(user_comment)
        print(f"Computed embedding user comment: {user_comment_embedding}")

        recommendations = []

        for comment in comments:
            similarity = self.similarity_model.compute_similarity(comment.embedding, user_comment_embedding)
            recommendations.append([comment.text, similarity])

        print("Computed similarity recommendations")
        print(recommendations[0])

        recommendations.sort(key=lambda x: x[1], reverse=True)

        print("Sorted recommendations")

        recommendations = recommendations[:3]
        random.shuffle(recommendations)

        return [recommendation[0] for recommendation in recommendations]
