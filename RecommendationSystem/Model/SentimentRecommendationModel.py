import time
from typing import Any

from RecommendationSystem.DB.utils import get_sentiment_comments_for_given_clusters
from RecommendationSystem.Model.RecommendationModel import RecommendationModel


class SentimentRecommendationModel(RecommendationModel):
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

        relevant_sentiment_comments = self.find_relevant_sentiment_comments(user_comment, cluster)

        print(f"Ping Sentiment: {time.time() - start}")

        return relevant_sentiment_comments

    def find_relevant_sentiment_comments(self, user_comment, cluster):
        positive_comments, neutral_comments, negative_comments = get_sentiment_comments_for_given_clusters(cluster)
        user_comment_embedding = self.embedding_model.embed_texts(user_comment)
        relevant_positive_comments = self.extract_relevant_comments(user_comment_embedding, positive_comments)
        relevant_neutral_comments = self.extract_relevant_comments(user_comment_embedding, neutral_comments)
        relevant_negative_comments = self.extract_relevant_comments(user_comment_embedding, negative_comments)
        relevant_sentiment_comments = self.mix_comments(user_comment, relevant_positive_comments, relevant_neutral_comments,
                                                        relevant_negative_comments)
        return relevant_sentiment_comments
