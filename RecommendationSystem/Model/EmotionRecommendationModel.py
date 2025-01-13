import time
from typing import Any

from RecommendationSystem.DB.utils import get_emotion_comments_for_given_clusters
from RecommendationSystem.Model.RecommendationModel import RecommendationModel


class EmotionRecommendationModel(RecommendationModel):
    """
    Recommendation model to extract the recommendations from the database.
    """

    def __init__(self, similarity_model, embedding_model):
        super().__init__(similarity_model, embedding_model)

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
        start = time.time()
        topics = self.find_topics(keywords_embeddings)

        print(f"Ping Topic: {time.time() - start}")

        articles = self.find_all_suitable_article(keywords_embeddings, topics)

        print(f"Ping Article: {time.time() - start}")

        cluster = self.find_fitting_cluster(articles, user_comment)

        print(f"Ping Cluster: {time.time() - start}")

        relevant_emotion_comments = self.find_relevant_emotion_comments(user_comment, cluster)

        print(f"Ping Sentiment: {time.time() - start}")

        return relevant_emotion_comments

    def find_relevant_emotion_comments(self, user_comment, cluster):
        love_comments, fear_comments, anger_comments, joy_comments, surprise_comments, sadness_comments = \
            get_emotion_comments_for_given_clusters(cluster)

        user_comment_embedding = self.embedding_model.embed_texts(user_comment)

        relevant_love_comments = self.extract_relevant_comments(user_comment_embedding, love_comments)
        relevant_fear_comments = self.extract_relevant_comments(user_comment_embedding, fear_comments)
        relevant_anger_comments = self.extract_relevant_comments(user_comment_embedding, anger_comments)
        relevant_joy_comments = self.extract_relevant_comments(user_comment_embedding, joy_comments)
        relevant_surprise_comments = self.extract_relevant_comments(user_comment_embedding, surprise_comments)
        relevant_sadness_comments = self.extract_relevant_comments(user_comment_embedding, sadness_comments)

        relevant_emotion_comments = self.mix_comments(user_comment, relevant_love_comments, relevant_fear_comments,
                                                      relevant_anger_comments, relevant_joy_comments,
                                                      relevant_surprise_comments, relevant_sadness_comments)
        return relevant_emotion_comments
