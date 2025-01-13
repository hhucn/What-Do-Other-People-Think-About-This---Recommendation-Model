# Define your item pipelines here
#
# Don't forget to add your pipeline to the ITEM_PIPELINES setting
# See: https://docs.scrapy.org/en/latest/topics/item-pipeline.html


# useful for handling different item types with a single interface
import logging
import os

from neomodel import config

from RecommendationSystem.Clustering.Clustering import Clustering
from RecommendationSystem.DB.db_models.StanceDetection.ContraComment import ContraComment
from RecommendationSystem.DB.db_models.StanceDetection.ProComment import ProComment
from RecommendationSystem.DB.db_models.article import Article
from RecommendationSystem.DB.db_models.comment import Comment
from scrapy import Item

from RecommendationSystem.DB.db_models.topic import Topic
from RecommendationSystem.Embedder.embedding_model import EmbeddingModel
from RecommendationSystem.SemanticSimilarity.SemanticSimilarity import SemanticSimilarity
from RecommendationSystem.StanceDetection.DistillationModel.StanceDetection import StanceDetectionModel
from RecommendationSystem.StanceDetection.Stance import Stance

config.DATABASE_URL = os.environ.get("NEO4J_BOLT_URL", 'bolt://neo4j:test@neo4j:7687')

#mgp = MovieGroupProcess(K=10, alpha=0.01, beta=0.01, n_iters=30)


class ScraperPipeline(object):
    def __init__(self):
        self.sematic_similarity: SemanticSimilarity = SemanticSimilarity()
        self.embedding_model: EmbeddingModel = EmbeddingModel()
        self.clustering_model = Clustering(self.embedding_model)

    def process_item(self, item, spider) -> Item:
        """
        Stores the articles and comments in the database
        This method is called every time an item is processed
        :param item: Article item with information about the article and the comments that are published under the
        article
        :param spider:
        :return: Unmodified item. The data are only extracted and stored in the database.
        """
        logging.info("Process item")

        article = self.get_or_create_article_node(item)

        self.create_and_connect_topic_node_with_article_node(article, item)

        if "comments" not in item.keys():
            return item

        comments = self.create_comment_nodes(item)

        self.clustering_model.create_cluster_nodes_for_article_with_comments(article, comments)

        return item

    def create_and_connect_topic_node_with_article_node(self, article, item):
        for keyword in item["keywords"][0].split(","):
            topic: Topic = Topic.nodes.get_or_none(topic_name=keyword)
            keyword = keyword.replace("'", "")
            if topic is None:
                topic = Topic(
                    topic_name=keyword,
                    embedding=self.embedding_model.embed_texts(keyword).tolist()
                ).save()
            article.topic.connect(topic)

    def get_or_create_article_node(self, item):
        article: Article = Article.nodes.get_or_none(article_title=item["article_title"][0])
        if article is None:
            embeddings = self.embedding_model.embed_texts(item["keywords"][0]).tolist()
            article = Article(
                article_title=item["article_title"][0],
                keywords=item["keywords"],
                news_agency=item["news_agency"][0],
                pub_date=item["pub_date"][0],
                url=item["url"][0],
                embedding=embeddings
            ).save()
        return article

    def create_comment_nodes(self, item):
        comments = []
        for comment_text in item["comments"]:
            comment: Comment = Comment.nodes.get_or_none(text=comment_text)
            if comment is None:
                comment = Comment(
                    text=comment_text
                ).save()
            comments.append(comment)
        return comments

