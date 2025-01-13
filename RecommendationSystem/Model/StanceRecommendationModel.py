import os
import time
from typing import Any

from RecommendationSystem.DB.utils import get_stance_comments_for_given_clusters
from RecommendationSystem.Model.RecommendationModel import RecommendationModel


class StanceRecommendationModel(RecommendationModel):
    """
    Recommendation model to extract the recommendations from the database.
    """

    os.environ["TOKENIZERS_PARALLELISM"] = "true"

    def __init__(self, similarity_model, embedding_model):
        super().__init__(similarity_model, embedding_model)
        self.start = time.time()

    def get_recommendations(self, comment_data: dict) -> list[Any] | tuple[list[Any], list[Any], list[Any], list[Any]]:
        """
        Interface method for the REST API view
        :param comment_data: Dict with all information the model needs to extract the recommendations from the database
        :return: List of recommendations
        """

        # Add model here
        keywords, user_comment = self.extract_user_comment_and_keywords(comment_data)
        keywords_embeddings = [self.embedding_model.embed_texts(keyword) for keyword in keywords]

        print("Start retrieval")
        topics = self.find_topics(keywords_embeddings)

        print(f"Ping Topic: {time.time() - self.start}")

        articles = self.find_all_suitable_article(keywords_embeddings, topics)

        print(f"Ping Article: {time.time() - self.start}")

        cluster = self.find_fitting_cluster(articles, user_comment)

        print(f"Ping Cluster: {time.time() - self.start}")

        relevant_stance_comments = self.find_relevant_stance_comments(user_comment, cluster)

        print(f"Ping Stance: {time.time() - self.start}")

        return relevant_stance_comments

    def find_relevant_stance_comments(self, user_comment, clusters):
        pro_comments_from_cluster, contra_comments_for_given_cluster = \
            get_stance_comments_for_given_clusters(clusters)

        print(f"Got Comments from cluster {time.time() - self.start}")
        user_comment_embedding = self.embedding_model.embed_texts(user_comment)
        print(f"Computed user comment embedding {time.time() - self.start}")

        relevant_pro_comments = StanceRecommendationModel.extract_relevant_comments(user_comment_embedding, pro_comments_from_cluster)
        print(f"Got relevant pro comments {time.time() - self.start}")
        relevant_contra_comments = StanceRecommendationModel.extract_relevant_comments(user_comment_embedding, contra_comments_for_given_cluster)
        print(f"Got relevant contra comments {time.time() - self.start}")

        relevant_stance_comments = self.mix_comments(user_comment, relevant_pro_comments, relevant_contra_comments)
        print(f"Mixed comments {time.time() - self.start}")

        return relevant_stance_comments
