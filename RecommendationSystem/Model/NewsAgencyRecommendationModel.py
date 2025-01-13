import time
from typing import Any

from RecommendationSystem.DB.utils import get_news_agency_comments_for_given_cluster, \
    get_all_breitbart_article_from_given_topics, \
    get_all_ny_times_article_from_given_topics
from RecommendationSystem.Model.RecommendationModel import RecommendationModel


class NewsAgencyRecommendationModel(RecommendationModel):
    """
    Recommendation model to extract the recommendations from the database.
    """

    def __init__(self, similarity_model, embedding_model):
        super().__init__(similarity_model, embedding_model)
        self.start = 0

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
        self.start = time.time()
        topics = self.find_topics(keywords_embeddings)

        print(f"Ping Article: {time.time() - self.start}")

        relevant_news_agencies_comments = self.find_relevant_news_agency_comments(keywords_embeddings, topics, user_comment)

        print(f"Ping News Agencies: {time.time() - self.start}")

        return relevant_news_agencies_comments

    def find_news_agencies_comment_candidates(self, keywords_embeddings, topics, user_comment):
        print(f"Get ny times article from given topics: {time.time() - self.start}")
        suitable_new_york_times_article = self.find_suitable_ny_times_article(keywords_embeddings, topics)
        print(f"Get breitbart article from given topics: {time.time() - self.start}")
        suitable_breitbart_article = self.find_suitable_breitbart_article(keywords_embeddings, topics)

        print(f"Find NyTimes cluster:  {time.time() - self.start}")
        ny_times_suitable_cluster = self.find_fitting_cluster(suitable_new_york_times_article, user_comment)
        print(f"Find Breitbart cluster:  {time.time() - self.start}")
        suitable_breitbart_cluster = self.find_fitting_cluster(suitable_breitbart_article, user_comment)

        print(f"Get comments for given cluster: {time.time() - self.start}")
        ny_times_comments, breitbart_comments = get_news_agency_comments_for_given_cluster(ny_times_suitable_cluster,
                                                                                           suitable_breitbart_cluster)
        print(f"Got comments for given cluster: {time.time() - self.start}")

        return breitbart_comments, ny_times_comments

    def find_relevant_news_agency_comments(self, keywords_embeddings, topics, user_comment):
        breitbart_comments, ny_times_comments = self.find_news_agencies_comment_candidates(keywords_embeddings,
                                                                                           topics,
                                                                                           user_comment)

        user_comment_embedding = self.embedding_model.embed_texts(user_comment)

        relevant_comments_ny_times = self.extract_relevant_comments(user_comment_embedding, ny_times_comments)
        relevant_comments_breitbart = self.extract_relevant_comments(user_comment_embedding, breitbart_comments)
        relevant_news_agencies_comments = self.mix_comments(user_comment, relevant_comments_ny_times, relevant_comments_breitbart)
        return relevant_news_agencies_comments

    def find_suitable_breitbart_article(self, keywords_embeddings, topics):
        breitbart_article = get_all_breitbart_article_from_given_topics(topics)
        if len(breitbart_article) > 0:
            suitable_breitbart_article = self.find_most_suitable_article(keywords_embeddings, breitbart_article)
            return suitable_breitbart_article

        return []

    def find_suitable_ny_times_article(self, keywords_embeddings, topics):
        new_york_times_article = get_all_ny_times_article_from_given_topics(topics)
        if len(new_york_times_article) > 0:
            suitable_new_york_times_article = self.find_most_suitable_article(keywords_embeddings, new_york_times_article)
            return suitable_new_york_times_article

        return []

    @staticmethod
    def extract_user_comment_and_keywords(comment_data):
        keywords = [comment.strip() for comment in
                    comment_data["keywords"].replace("[", "").replace("]", "").split(",")]
        user_comment = comment_data["user_comment"]
        return keywords, user_comment
