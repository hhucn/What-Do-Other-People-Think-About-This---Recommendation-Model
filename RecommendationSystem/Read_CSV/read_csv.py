import csv
import logging
import os
import re
import subprocess
from collections import defaultdict
from datetime import datetime
from typing import Dict, List

from neomodel import config

from RecommendationSystem.Clustering.Clustering import Clustering
from RecommendationSystem.DB.db_models.EmotionClassification.AngerComment import AngerComment
from RecommendationSystem.DB.db_models.EmotionClassification.FearComment import FearComment
from RecommendationSystem.DB.db_models.EmotionClassification.JoyComment import JoyComment
from RecommendationSystem.DB.db_models.EmotionClassification.LoveComment import LoveComment
from RecommendationSystem.DB.db_models.EmotionClassification.SadnessComment import SadnessComment
from RecommendationSystem.DB.db_models.EmotionClassification.SurpriseComment import SurpriseComment
from RecommendationSystem.DB.db_models.StanceDetection.ContraComment import ContraComment
from RecommendationSystem.DB.db_models.SentimentClassification.NegativeComment import NegativeComment
from RecommendationSystem.DB.db_models.SentimentClassification.NeutralComment import NeutralComment
from RecommendationSystem.DB.db_models.SentimentClassification.PositiveComment import PositiveComment
from RecommendationSystem.DB.db_models.StanceDetection.ProComment import ProComment
from RecommendationSystem.DB.db_models.article import Article
from RecommendationSystem.DB.db_models.comment import Comment
from RecommendationSystem.DB.db_models.topic import Topic
from RecommendationSystem.Embedder.embedding_model import EmbeddingModel
from RecommendationSystem.EmotionClassification.EmotionClassification import EmotionClassification
from RecommendationSystem.EmotionClassification.EmotionEnum import Emotion
from RecommendationSystem.SemanticSimilarity.SemanticSimilarity import SemanticSimilarity
from RecommendationSystem.SentimentClassification.SentimentClassification import SentimentClassification
from RecommendationSystem.SentimentClassification.SentimentEnum import Sentiment
from RecommendationSystem.StanceDetection.DistillationModel.StanceDetection import StanceDetectionModel
from RecommendationSystem.StanceDetection.Stance import Stance
from RecommendationSystem.relevance_score.RelevanceScore import RelevanceScore

config.DATABASE_URL = os.environ.get("NEO4J_BOLT_URL", 'bolt://neo4j:fjowiejfiwej342j349j@localhost:7687')


class ReadCSV:
    def __init__(self, relevance_score, stance_detection_model, semantic_similarity_model, embedding_model,
                 sentiment_classification_model, emotion_classification_model):
        self.relevance_score: RelevanceScore = relevance_score
        self.emotion_classification = emotion_classification_model
        self.sentiment_classification = sentiment_classification_model
        self.stance_model = stance_detection_model
        self.similarity_model = semantic_similarity_model
        self.embedding_model = embedding_model
        self.article_nodes = {}
        self.comment_nodes = defaultdict(list)
        self.clustering_model = Clustering(self.embedding_model)

    def __create_and_connect_topic_nodes_with_article_node(self, article_node: Article, keywords: List[str]):
        for keyword in keywords:
            topic: Topic = Topic.nodes.get_or_none(topic_name=keyword)
            if topic is None:
                topic_embedding = self.embedding_model.embed_texts(keyword).tolist()
                topic = Topic(
                    topic_name=keyword.lower(),
                    embedding=topic_embedding
                ).save()
            article_node.topic.connect(topic)

    def store_articles_in_db(self, article_file: str, news_agency: str) -> None:
        """
        Reads the give article file and stores them in the db
        :param article_file:
        :param news_agency:
        :return:
        """
        with open(article_file, encoding="utf-8-sig") as csv_file:
            csv_reader = csv.DictReader(csv_file)
            articles = list(csv_reader)
            article_number = len(articles)
            for processed_number, node in enumerate(articles):
                try:
                    article_node, keywords = self.__store_article(node, news_agency)
                    self.__create_and_connect_topic_nodes_with_article_node(article_node, keywords)
                except KeyError:
                    pass
                if processed_number % 20 == 0:
                    logging.info(f"Articles processed: {processed_number} / {article_number} \n")

    def __store_article(self, node: Dict, news_agency: str) -> [Article, List[str]]:
        """
        Stores the give node in the db
        :param article: Dict with article information
        :return:
        """
        # Save the node in the db like this
        article = Article.nodes.get_or_none(article_id=node["articleID"])
        keywords = self.parse_keywords(node, news_agency)

        if article is None:
            keyword_embeddings = self.embedding_model.embed_texts(str(keywords)).tolist()
            article = Article(
                article_id=node['articleID'],
                article_title=node['headline'],
                news_agency=news_agency,
                keywords=str(keywords),
                pub_date=datetime.today().date(),
                url=node['webURL'],
                embedding=keyword_embeddings
            ).save()
            self.article_nodes[node['articleID']] = article
            return article, keywords
        self.article_nodes[node['articleID']] = article
        return article, keywords

    @staticmethod
    def parse_keywords(article, news_agency):
        if "keywords" not in article.keys():
            return ""
        keywords = article['keywords'].replace("[", "")
        keywords = keywords.replace("]", "")
        if news_agency == 'Breitbart':
            keywords = keywords.split(",")
        else:
            keywords = keywords.split("\'")
        keywords = [s for s in keywords if s != ', ' and len(s) != 0]
        keywords = [keyword.lower() for keyword in keywords]
        return keywords

    def store_comments_in_db(self, comment_full_path: str):
        with open(comment_full_path, encoding="utf-8-sig") as csv_file:
            csv_reader = csv.DictReader(csv_file)
            comments = list(csv_reader)
            comment_number = len(comments)
            for processed_comments, node in enumerate(comments):
                try:
                    self.store_comment(node)
                except KeyError:
                    pass
                if processed_comments % 500 == 0:
                    logging.info(f"Comments processed: {processed_comments} / {comment_number} \n")

    def store_comment(self, node: dict):
        comment_text = self.preprocess_comment(node['commentBody'])
        comment_id = node['commentID']
        comment = Comment.nodes.get_or_none(comment_id=comment_id)
        article = Article.nodes.get_or_none(article_id=node['articleID'])
        comment_embedding = self.embedding_model.embed_texts(comment_text).tolist()

        if article is not None and len(article.keywords) > 0 and len(comment_text) > 0 and comment is None:
            relevance_score = self.relevance_score.compute_relevance_score(comment_text)
            reason_score = self.relevance_score.compute_reason_statement_score(comment_text)
            personal_story_score = self.relevance_score.compute_personal_story_score(comment_text)
            example_score = self.relevance_score.compute_example_score(comment_text)
            source_score = self.relevance_score.compute_source_score(comment_text)

            stance = self.compute_stance_for_comment(article, comment_text)
            comment = self.store_stance_comment(comment, comment_embedding, comment_text, stance,
                                                relevance_score, reason_score, personal_story_score, example_score, source_score)

            self.comment_nodes[node['articleID']].append(comment)

            sentiment = self.sentiment_classification.compute_sentiment(comment_text)

            comment = self.store_sentiment_comment(comment, comment_embedding, comment_text, sentiment,
                                                   relevance_score, reason_score,
                                                   personal_story_score, example_score, source_score)

            self.comment_nodes[node['articleID']].append(comment)

            emotion = self.emotion_classification.compute_emotion(comment_text)

            comment = self.store_emotion_comment(comment, comment_embedding, comment_text, emotion, relevance_score, reason_score,
                                                 personal_story_score, example_score, source_score)

            self.comment_nodes[node['articleID']].append(comment)

    def store_emotion_comment(self, comment, comment_embedding, comment_text, emotion, relevance_score, reason_score,
                              personal_story_score, example_score, source_score):
        if emotion is Emotion.LOVE:
            comment = self.__store_love_comment_in_db(comment_text, comment_embedding, relevance_score, reason_score,
                                                      personal_story_score, example_score, source_score)
        elif emotion is Emotion.FEAR:
            comment = self.__store_fear_comment_in_db(comment_text, comment_embedding, relevance_score, reason_score,
                                                      personal_story_score, example_score, source_score)
        elif emotion is Emotion.ANGER:
            comment = self.__store_anger_comment_in_db(comment_text, comment_embedding, relevance_score, reason_score,
                                                       personal_story_score, example_score, source_score)
        elif emotion is Emotion.JOY:
            comment = self.__store_joy_comment_in_db(comment_text, comment_embedding, relevance_score, reason_score,
                                                     personal_story_score, example_score, source_score)
        elif emotion is Emotion.SURPRISE:
            comment = self.__store_surprise_comment_in_db(comment_text, comment_embedding, relevance_score, reason_score,
                                                          personal_story_score, example_score, source_score)
        elif emotion is Emotion.SADNESS:
            comment = self.__store_sadness_comment_in_db(comment_text, comment_embedding, relevance_score, reason_score,
                                                         personal_story_score, example_score, source_score)
        return comment

    def store_sentiment_comment(self, comment, comment_embedding, comment_text, sentiment, relevance_score, reason_score,
                                personal_story_score, example_score, source_score):
        if sentiment is Sentiment.POSITIVE:
            comment = self.__store_positive_comment_in_db(comment_text, comment_embedding, relevance_score, reason_score,
                                                          personal_story_score, example_score, source_score)
        elif sentiment is Sentiment.NEUTRAL:
            comment = self.__store_neutral_comment_in_db(comment_text, comment_embedding, relevance_score, reason_score,
                                                         personal_story_score, example_score, source_score)
        elif sentiment is Sentiment.NEGATIVE:
            comment = self.__store_negative_comment_in_db(comment_text, comment_embedding, relevance_score, reason_score,
                                                          personal_story_score, example_score, source_score)
        return comment

    def store_stance_comment(self, comment, comment_embedding, comment_text, stance, relevance_score, reason_score,
                             personal_story_score, example_score, source_score):
        if stance is Stance.PRO:
            comment = self.__store_pro_comment_in_db(comment_text, comment_embedding, relevance_score, reason_score,
                                                     personal_story_score, example_score, source_score)
        elif stance is Stance.CONTRA:
            comment = self.__store_contra_comment_in_db(comment_text, comment_embedding, relevance_score, reason_score,
                                                        personal_story_score, example_score, source_score)
        return comment

    def compute_stance_for_comment(self, article, comment_text):
        stance = None

        # ML model is trained on specific keywords e.g. nuclear energy. Related topics like global warming need to be maped onto these
        # compute stance correctly
        if "trump" in article.keywords.lower():
            stance = self.stance_model.compute_stance("Donald Trump", comment_text)
        elif 'global warming' in article.keywords.lower():
            stance = self.stance_model.compute_stance("nuclear energy", comment_text)
        elif 'abortion' in article.keywords.lower():
            stance = self.stance_model.compute_stance("abortion", comment_text)
        else:
            stance = self.stance_model.compute_stance(article.keywords, comment_text)
        return stance

    @staticmethod
    def __store_pro_comment_in_db(comment_text, comment_embedding, relevance_score, reason_score,
                                  personal_story_score, example_score, source_score):
        comment = ProComment(
            text=comment_text,
            embedding=comment_embedding,
            relevance_score=relevance_score,
            reason_score=reason_score,
            personal_story_score=personal_story_score,
            example_score=example_score,
            source_score=source_score
        ).save()
        return comment

    @staticmethod
    def __store_contra_comment_in_db(comment_text, comment_embedding, relevance_score, reason_score,
                                     personal_story_score, example_score, source_score):
        comment = ContraComment(
            text=comment_text,
            embedding=comment_embedding,
            relevance_score=relevance_score,
            reason_score=reason_score,
            personal_story_score=personal_story_score,
            example_score=example_score,
            source_score=source_score
        ).save()
        return comment

    # Code from https://www.slingacademy.com/article/python-ways-to-remove-html-tags-from-a-string/#Using_HTMLParser Accessed: 17.08.2023
    @staticmethod
    def preprocess_comment(text):
        clean = re.compile('<.*?>')
        return re.sub(clean, '', text)

    @staticmethod
    def __store_positive_comment_in_db(comment_text, comment_embedding, relevance_score, reason_score, personal_story_score, example_score, source_score):
        comment = PositiveComment(
            text=comment_text,
            embedding=comment_embedding,
            relevance_score=relevance_score,
            reason_score=reason_score,
            personal_story_score=personal_story_score,
            example_score=example_score,
            source_score=source_score
        ).save()
        return comment

    @staticmethod
    def __store_neutral_comment_in_db(comment_text, comment_embedding, relevance_score, reason_score,
                                      personal_story_score, example_score, source_score):
        comment = NeutralComment(
            text=comment_text,
            embedding=comment_embedding,
            relevance_score=relevance_score,
            reason_score=reason_score,
            personal_story_score=personal_story_score,
            example_score=example_score,
            source_score=source_score
        ).save()
        return comment

    @staticmethod
    def __store_negative_comment_in_db(comment_text, comment_embedding, relevance_score, reason_score,
                                       personal_story_score, example_score, source_score):
        comment = NegativeComment(
            text=comment_text,
            embedding=comment_embedding,
            relevance_score=relevance_score,
            reason_score=reason_score,
            personal_story_score=personal_story_score,
            example_score=example_score,
            source_score=source_score
        ).save()
        return comment

    @staticmethod
    def __store_love_comment_in_db(comment_text, comment_embedding, relevance_score, reason_score,
                                   personal_story_score, example_score, source_score):
        comment = LoveComment(
            text=comment_text,
            embedding=comment_embedding,
            relevance_score=relevance_score,
            reason_score=reason_score,
            personal_story_score=personal_story_score,
            example_score=example_score,
            source_score=source_score
        ).save()
        return comment

    @staticmethod
    def __store_fear_comment_in_db(comment_text, comment_embedding, relevance_score, reason_score,
                                   personal_story_score, example_score, source_score):
        comment = FearComment(
            text=comment_text,
            embedding=comment_embedding,
            relevance_score=relevance_score,
            reason_score=reason_score,
            personal_story_score=personal_story_score,
            example_score=example_score,
            source_score=source_score
        ).save()
        return comment

    @staticmethod
    def __store_anger_comment_in_db(comment_text, comment_embedding, relevance_score, reason_score,
                                    personal_story_score, example_score, source_score):
        comment = AngerComment(
            text=comment_text,
            embedding=comment_embedding,
            relevance_score=relevance_score,
            reason_score=reason_score,
            personal_story_score=personal_story_score,
            example_score=example_score,
            source_score=source_score
        ).save()
        return comment

    @staticmethod
    def __store_joy_comment_in_db(comment_text, comment_embedding, relevance_score, reason_score,
                                  personal_story_score, example_score, source_score):
        comment = JoyComment(
            text=comment_text,
            embedding=comment_embedding,
            relevance_score=relevance_score,
            reason_score=reason_score,
            personal_story_score=personal_story_score,
            example_score=example_score,
            source_score=source_score
        ).save()
        return comment

    @staticmethod
    def __store_surprise_comment_in_db(comment_text, comment_embedding, relevance_score, reason_score,
                                       personal_story_score, example_score, source_score):
        comment = SurpriseComment(
            text=comment_text,
            embedding=comment_embedding,
            relevance_score=relevance_score,
            reason_score=reason_score,
            personal_story_score=personal_story_score,
            example_score=example_score,
            source_score=source_score
        ).save()
        return comment

    @staticmethod
    def __store_sadness_comment_in_db(comment_text, comment_embedding, relevance_score, reason_score,
                                      personal_story_score, example_score, source_score):
        comment = SadnessComment(
            text=comment_text,
            embedding=comment_embedding,
            relevance_score=relevance_score,
            reason_score=reason_score,
            personal_story_score=personal_story_score,
            example_score=example_score,
            source_score=source_score
        ).save()
        return comment


def main() -> None:
    """
    Store all data from csv in db
    :return: None
    """

    logging.basicConfig(level=logging.INFO)

    relevance_score = RelevanceScore()
    stance_detection_model = StanceDetectionModel()
    semantic_similarity_model = SemanticSimilarity()
    embedding_model = EmbeddingModel()
    sentiment_classification = SentimentClassification()
    emotion_classification = EmotionClassification()
    read_csv = ReadCSV(relevance_score, stance_detection_model, semantic_similarity_model, embedding_model, sentiment_classification,
                       emotion_classification)

    article_file_names = [["Breitbart/Breitbart_Articles_Trump_Abortion_ClimateChange.csv", "Breitbart"]]
    comment_file_names = ["Breitbart/Breitbart_Comments_Trump_Abortion_ClimateChange.csv"]

    # Make a list of all csv files you would like to store in the db like this
    absolute_path = os.path.dirname(__file__)
    relative_path = "data/"

    for article_file_name, news_agency in article_file_names:
        article_full_path = os.path.join(absolute_path, relative_path + article_file_name)
        read_csv.store_articles_in_db(article_full_path, news_agency)

    for comment_file_name in comment_file_names:
        comment_full_path = os.path.join(absolute_path, relative_path + comment_file_name)
        read_csv.store_comments_in_db(comment_full_path)

    for article_id in read_csv.article_nodes.keys():
        article = read_csv.article_nodes.get(article_id)
        comments = read_csv.comment_nodes.get(article_id)
        if article is not None and comments is not None:
            read_csv.clustering_model.create_cluster_nodes_for_article_with_comments(article, comments)

    logging.info("Import into DB done")


def set_alias():
    set_alias_command = f"mc alias set cn-s3 https://s3.cs.hhu.de {os.getenv('s3_user')} {os.getenv('s3_password')}"
    set_alias_process = subprocess.Popen(set_alias_command, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)

    process_stdout, process_stderr = set_alias_process.communicate()

    if set_alias_process.returncode != 0:
        print(f"Command failed with error: {process_stderr.decode()}")
    else:
        print("Setting alias succeeded")


if __name__ == '__main__':
    set_alias()
    copy_csv_files_command = "mc cp --recursive cn-s3/comment-recommendation/ /code/RecommendationSystem/Read_CSV/data"
    process = subprocess.Popen(copy_csv_files_command, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    stdout, stderr = process.communicate()
    if process.returncode != 0:
        print(f"Command failed with error: {stderr.decode()}")
    else:
        print("Command succeeded")
        main()
