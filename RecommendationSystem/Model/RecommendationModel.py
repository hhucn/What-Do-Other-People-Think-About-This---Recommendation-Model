import random
from typing import List

import numpy as np
import torch
from numpy.linalg import norm
from sentence_transformers import util
from torch import Tensor

from RecommendationSystem.API.RESTApi.abstract_model import AbstractModel
from RecommendationSystem.Clustering.Clustering import Clustering
from RecommendationSystem.DB.db_models.article import Article
from RecommendationSystem.DB.db_models.cluster import Cluster
from RecommendationSystem.DB.db_models.topic import Topic
from RecommendationSystem.DB.utils import get_entry_level_topics, get_all_cluster_from_given_articles, get_all_article_from_given_topics
from RecommendationSystem.Embedder.embedding_model import EmbeddingModel
from RecommendationSystem.SemanticSimilarity import SemanticSimilarity


class RecommendationModel(AbstractModel):
    def __init__(self, similarity_model, embedding_model):
        self.number_of_article = 10
        self.embedding_model: EmbeddingModel = embedding_model
        self.similarity_model: SemanticSimilarity = similarity_model
        self.topic_similarity_threshold = 0.70
        self.article_similarity_threshold = 0.6
        self.cluster_similarity_threshold = 0.4
        self.number_of_topics = 20
        self.number_of_comments = 3
        self.entry_level_topics_array = np.array(get_entry_level_topics())
        self.topic_embeddings_array = np.array([t.embedding for t in self.entry_level_topics_array]).astype('float32')

        self.clustering_model = Clustering(self.embedding_model)

    def find_fitting_cluster(self, suitable_articles, user_comment):
        cluster_for_given_article = np.array(get_all_cluster_from_given_articles(suitable_articles))
        if len(cluster_for_given_article) > 0:
            suitable_cluster = self.get_cluster_for_user_comment(user_comment, cluster_for_given_article,
                                                                 self.cluster_similarity_threshold)
            return suitable_cluster

        return []

    def find_all_suitable_article(self, keywords_embeddings, topics):
        articles = get_all_article_from_given_topics(topics)
        suitable_articles = self.find_most_suitable_article(keywords_embeddings, articles)
        return suitable_articles

    @staticmethod
    def extract_user_comment_and_keywords(comment_data):
        keywords = [comment.strip() for comment in
                    comment_data["keywords"].replace("[", "").replace("]", "").split(",")]
        user_comment = comment_data["user_comment"]
        return keywords, user_comment

    def find_most_suitable_article(self, keywords_embeddings: List[Tensor], articles: List[Article]) -> List[Article]:
        article_embeddings = np.array([article.embedding for article in articles]).astype('float32')
        average_keyword_embedding = torch.mean(torch.stack(keywords_embeddings), dim=0)

        similarities = util.cos_sim(average_keyword_embedding, article_embeddings)[0]

        article_candidates = list(zip(articles, similarities))

        article_candidates.sort(key=lambda x: x[1], reverse=True)
        return [article_candidate[0] for article_candidate in article_candidates][:self.number_of_article]

    def get_cluster_for_user_comment(self, user_comment: str, clusters: List[Cluster], threshold: float) -> \
            List[Article]:
        preprocessed_user_comment = self.clustering_model.preprocess_comments_for_clustering([user_comment])[0]
        user_comment_embedding = self.embedding_model.embed_vector(preprocessed_user_comment)

        cluster_embeddings = np.array([cluster.embedding for cluster in clusters]).astype('float32')
        matching_cluster_indices = np.flatnonzero(util.cos_sim(user_comment_embedding, cluster_embeddings) > threshold)
        matching_cluster = clusters[matching_cluster_indices]

        return matching_cluster

    @staticmethod
    def extract_relevant_comments(user_comment_embedding, comments_from_cluster):
        embeddings = np.empty((len(comments_from_cluster), len(comments_from_cluster[0].embedding)))
        relevance_scores = np.empty(len(comments_from_cluster))
        user_comment_embedding = np.array(user_comment_embedding).astype("float64")

        for i in range(len(comments_from_cluster)):
            embeddings[i] = comments_from_cluster[i].embedding
            relevance_scores[i] = comments_from_cluster[i].relevance_score

        similarities = (np.dot(embeddings, user_comment_embedding) /
                        (norm(embeddings, axis=1) * norm(user_comment_embedding)))
        weighted_relevance_scores = similarities * relevance_scores
        relevant_comments = list(zip(comments_from_cluster, weighted_relevance_scores))

        relevant_comments.sort(key=lambda x: x[1], reverse=True)

        return [t[0].text for t in relevant_comments][:10]

    def find_topics(self, keywords_embeddings: List[float]) -> List[Topic]:
        suitable_topics = []

        for keyword_embedding in keywords_embeddings:
            indices = np.flatnonzero(util.cos_sim(keyword_embedding, self.topic_embeddings_array) > self.topic_similarity_threshold)
            topics = self.entry_level_topics_array[indices]
            suitable_topics.extend(topics)

        return suitable_topics

    def mix_comments(self, user_comment, *comment_lists):
        random.seed(10)
        comments = []

        for comments_list in comment_lists:
            comments.extend([comment for comment in comments_list if comment != user_comment][:2])

        random.shuffle(comments)

        return comments[:self.number_of_comments]
