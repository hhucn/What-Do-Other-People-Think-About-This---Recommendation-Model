import os
import time
from typing import List

import torch
from neo4j import GraphDatabase
from neomodel import config, db, StructuredNode

from RecommendationSystem.DB.db_models.EmotionClassification.AngerComment import AngerComment
from RecommendationSystem.DB.db_models.EmotionClassification.FearComment import FearComment
from RecommendationSystem.DB.db_models.EmotionClassification.JoyComment import JoyComment
from RecommendationSystem.DB.db_models.EmotionClassification.LoveComment import LoveComment
from RecommendationSystem.DB.db_models.EmotionClassification.SadnessComment import SadnessComment
from RecommendationSystem.DB.db_models.EmotionClassification.SurpriseComment import SurpriseComment
from RecommendationSystem.DB.db_models.SentimentClassification.NegativeComment import NegativeComment
from RecommendationSystem.DB.db_models.SentimentClassification.NeutralComment import NeutralComment
from RecommendationSystem.DB.db_models.SentimentClassification.PositiveComment import PositiveComment
from RecommendationSystem.DB.db_models.StanceDetection.ContraComment import ContraComment
from RecommendationSystem.DB.db_models.StanceDetection.ProComment import ProComment
from RecommendationSystem.DB.db_models.article import Article
from RecommendationSystem.DB.db_models.cluster import Cluster
from RecommendationSystem.DB.db_models.comment import Comment
from RecommendationSystem.DB.db_models.topic import Topic

config.DATABASE_URL = os.environ.get("NEO4J_BOLT_URL", 'bolt://neo4j:test@neo4j:7687')

driver = GraphDatabase.driver("bolt://neo4j", auth=('neo4j', os.environ.get('NEO4_PASSWORD')))


def __extract_results(results: List, node_type: StructuredNode) -> List:
    """
    Extracts the results from the query response
    :param results: Results from cypher query which should be inflated to nodes of type node_type
    :param node_type:
    :return: List with nodes of type node_type
    """
    return [node_type.inflate(row[0]) for row in results]


def get_article_by_title(title: str) -> Article:
    """
    Gets a specific article by the give title
    :param title: Title of the article
    :return: Article node with given title
    """
    results, _ = db.cypher_query(
        f"""
        MATCH (a:Article)
        WHERE a.article_title="{title}" AND a.embedding IS NOT NULL
        RETURN a
        """
    )
    if len(results) != 0:
        return __extract_results(results, Article)[0]
    return None


def get_article_by_id(article_id: int) -> Article:
    """
    Gets a specific article by the given id
    :param article_id: ID of article node
    :return: Article node with given ID
    """
    results, _ = db.cypher_query(
        f"""
        MATCH (a:Article)
        WHERE a.article_id='{article_id}' AND a.embedding IS NOT NULL
        RETURN a
        """
    )
    if len(results) != 0:
        return __extract_results(results, Article)[0]
    return None


def get_all_article() -> List:
    """
    Gets all article from the db which embedding is not null
    :return: All articles from the database whose embedding is not NULL
    """
    results, _ = db.cypher_query(
        """
        MATCH (a:Article)
        WHERE a.embedding IS NOT NULL
        RETURN a
        """
    )
    return __extract_results(results, Article)


def get_comment_by_id(comment_id: int) -> Comment:
    """
    Gets a specific comment by the given id
    :param comment_id: ID of the comment
    :return: Comment node with given ID
    """
    results, _ = db.cypher_query(
        f"""
        MATCH(c:Comment)
        WHERE c.comment_id='{comment_id}' AND c.embedding IS NOT NULL
        RETURN c
        """
    )
    if len(results) != 0:
        return __extract_results(results, Comment)[0]
    return None


def get_all_comments_for_given_article(article: Article) -> List[Comment]:
    """
    Returns all comments connected to the given article
    :param article: Article node
    :return: List of all comment nodes that are connected with the given article node
    """
    results, _ = db.cypher_query(
        f"""
        MATCH (a:Article)<--(k:Cluster)<--(c:Comment)
        WHERE a.article_id='{article.article_id}' AND c.embedding IS NOT NULL
        RETURN c
        """
    )
    return __extract_results(results, Comment)


def get_all_comments_by_article_id(article_id: int) -> List[Comment]:
    """
    Returns all comments connected to the given article
    :param article: Article node
    :return: List of all comment nodes that are connected with the given article node
    """
    results, _ = db.cypher_query(
        f"""
        MATCH (a:Article)<--(k:Cluster)<--(c:Comment)
        WHERE a.article_id='{article_id}' AND c.embedding IS NOT NULL
        RETURN c
        """
    )
    return __extract_results(results, Comment)


def run_cypher_query(query: str, node_type) -> List:
    """
    Runs a give cypher query to handle special cases
    :param query: Cypher query string to extract specific data from the Neo4J database
    :param node_type: Type of the expected node to be returned
    :return: List of nodes for the given cypher query
    """
    results, _ = db.cypher_query(query)
    return __extract_results(results, node_type)


def get_articles_without_embedding():
    """
    Gets all article without an embedding
    :return:
    """
    results, _ = db.cypher_query(
        """
        MATCH (a:Article)
        WHERE a.embedding IS NULL
        RETURN a
        """
    )

    return __extract_results(results, Article)


def get_comments_without_embedding():
    """
    Gets all comments without an embedding
    :return:
    """
    results, _ = db.cypher_query(
        """
        MATCH (c:Comment)
        WHERE c.embedding IS NULL
        RETURN c
        """
    )
    return __extract_results(results, Comment)


def update_node_with_embedding(node: StructuredNode, embedding: torch.Tensor):
    """
    Updates the given node with the given embedding
    :param node:
    :param embedding:
    :return:
    """
    node.embedding = embedding.tolist()
    node.save()


def get_entry_level_topics():
    results, _ = db.cypher_query(
        f"""
        MATCH (t:Topic)<--(a:Article)
        RETURN t, count(a) ORDER BY count(a) DESC
        """
    )
    return __extract_results(results, Topic)


def get_topics_for_given_article(article: Article) -> List[Topic]:
    results, _ = db.cypher_query(
        f"""
        MATCH (t:Topic)<--(a:Article)
        WHERE a.article_id="{article.article_id}"
        RETURN t
        """
    )

    return __extract_results(results, Topic)


def get_number_of_connections(topic: Topic) -> float:
    results, _ = db.cypher_query(
        f"""
        MATCH (t:Topic)<--(a:Article) 
        WHERE t.topic_id="{topic.topic_id}"
        RETURN count(a)
        """
    )

    return float(results[0][0])


def __get_all_cluster_from_given_article(article):
    results, _ = db.cypher_query(
        f"""
        MATCH (a:Article)<--(c:Cluster)
        WHERE a.article_id="{article.article_id}" AND a.embedding IS NOT NULL
        RETURN c
        """
    )

    return __extract_results(results, Cluster)


def __get_all_cluster_from_given_article_from_news_agency(article, new_agency):
    results, _ = db.cypher_query(
        f"""
        MATCH (a:Article)<--(c:Cluster)
        WHERE a.article_id="{article.article_id}" AND a.embedding IS NOT NULL AND a.news_agency="{new_agency}"
        RETURN c
        """
    )

    return __extract_results(results, Cluster)


def get_all_cluster_from_given_articles(articles: List[Article]) -> List[Cluster]:
    cluster = []

    for article in articles:
        cluster.extend(__get_all_cluster_from_given_article(article))

    return cluster


def get_all_article_from_given_topics(topics: List[Topic]) -> list[Article]:
    articles = []

    for topic in topics:
        articles.extend(__get_article_for_given_topics(topic))

    return list(set(articles))


def get_all_ny_times_article_from_given_topics(topics: List[Topic]) -> list[Article]:
    new_york_times_articles = []

    for topic in topics:
        new_york_times_articles.extend(__get_article_for_given_topics_and_news_agency(topic, "NewYorkTimes"))

    return list(set(new_york_times_articles))


def get_all_breitbart_article_from_given_topics(topics: List[Topic]) -> list[Article]:
    breitbart_articles = []

    for topic in topics:
        breitbart_articles.extend(__get_article_for_given_topics_and_news_agency(topic, "Breitbart"))

    return list(set(breitbart_articles))


def __get_article_for_given_topics(topic: Topic) -> List[Article]:
    results, _ = db.cypher_query(
        f"""
        MATCH (t:Topic)<--(a:Article)
        WHERE t.topic_id='{topic.topic_id}' and a.embedding IS NOT NULL
        RETURN a
        """
    )
    return __extract_results(results, Article)


def __get_article_for_given_topics_and_news_agency(topic: Topic, news_agency: str) -> List[Article]:
    results, _ = db.cypher_query(
        f"""
        MATCH (t:Topic)<--(a:Article)
        WHERE t.topic_id='{topic.topic_id}' and a.news_agency='{news_agency}' and a.embedding IS NOT NULL
        RETURN a
        """
    )
    return __extract_results(results, Article)


def create_query_for_pro_comments(clusters: List[Cluster]):
    where_clause = ""
    for cluster in clusters[:-1]:
        where_clause += f""" k.uid = '{cluster.uid}' or """

    where_clause += f""" k.uid = '{clusters[-1].uid}'"""

    return f"""
            MATCH (k:Cluster)<--(c:ProComment)
            WHERE {where_clause} and c.relevance_score > 0.7
            RETURN  c
            """


def create_query_for_contra_comments(clusters: List[Cluster]):
    where_clause = ""
    for cluster in clusters[:-1]:
        where_clause += f"""k.uid = '{cluster.uid}' or """

    where_clause += f"""k.uid = '{clusters[-1].uid}' """

    return f"""
             MATCH (k:Cluster)<--(c:ContraComment)
             WHERE {where_clause} and c.relevance_score > 0.7
             RETURN  c
             """


def process_cluster(tx, query, node_type, results):
    nodes = tx.run(query)

    for node in nodes:
        results.append(node_type.inflate(node["c"]))


def get_stance_comments_for_given_clusters(cluster: Cluster) -> List[ProComment]:
    results_pro_comments = []
    results_contra_comments = []

    pro_query = create_query_for_pro_comments(cluster)
    contra_query = create_query_for_contra_comments(cluster)

    with driver.session(database='neo4j') as session:
        session.execute_read(process_cluster, pro_query, ProComment, results_pro_comments)
        session.execute_read(process_cluster, contra_query, ContraComment, results_contra_comments)

    return results_pro_comments, results_contra_comments


def __get_contra_comments_for_given_cluster(cluster: Cluster) -> List[ContraComment]:
    results, _ = db.cypher_query(
        f"""
        MATCH (c:Cluster)<--(k:ContraComment)
        WHERE c.uid = '{cluster.uid}' and k.embedding IS NOT NULL and k.relevance_score > 0.7
        RETURN k
        """
    )
    return __extract_results(results, ContraComment)


def __get_comments_for_given_cluster(cluster: Cluster) -> List[Comment]:
    results, _ = db.cypher_query(
        f"""
        MATCH (c:Cluster)<--(k:Comment)
        WHERE c.uid = '{cluster.uid}' and k.embedding IS NOT NULL
        RETURN k
        """
    )
    return __extract_results(results, Comment)


def __get_positive_comments_for_given_cluster(cluster: Cluster) -> List[PositiveComment]:
    results, _ = db.cypher_query(
        f"""
            MATCH (c:Cluster)<--(k:PositiveComment)
            WHERE c.uid = '{cluster.uid}' and k.embedding IS NOT NULL
            RETURN k
            """
    )
    return __extract_results(results, PositiveComment)


def __get_neutral_comments_for_given_cluster(cluster: Cluster) -> List[NeutralComment]:
    results, _ = db.cypher_query(
        f"""
                MATCH (c:Cluster)<--(k:NeutralComment)
                WHERE c.uid = '{cluster.uid}' and k.embedding IS NOT NULL
                RETURN k
                """
    )
    return __extract_results(results, NeutralComment)


def __get_negative_comments_for_given_cluster(cluster: Cluster) -> List[NegativeComment]:
    results, _ = db.cypher_query(
        f"""
                MATCH (c:Cluster)<--(k:NegativeComment)
                WHERE c.uid = '{cluster.uid}' and k.embedding IS NOT NULL
                RETURN k
                """
    )
    return __extract_results(results, NegativeComment)


def __get_love_comments_for_given_cluster(cluster: Cluster) -> List[LoveComment]:
    results, _ = db.cypher_query(
        f"""
                MATCH (c:Cluster)<--(k:LoveComment)
                WHERE c.uid = '{cluster.uid}' and k.embedding IS NOT NULL
                RETURN k
                """
    )
    return __extract_results(results, LoveComment)


def __get_fear_comments_for_given_cluster(cluster: Cluster) -> List[FearComment]:
    results, _ = db.cypher_query(
        f"""
                    MATCH (c:Cluster)<--(k:FearComment)
                    WHERE c.uid = '{cluster.uid}' and k.embedding IS NOT NULL
                    RETURN k
                    """
    )
    return __extract_results(results, FearComment)


def __get_anger_comments_for_given_cluster(cluster: Cluster) -> List[AngerComment]:
    results, _ = db.cypher_query(
        f"""
                    MATCH (c:Cluster)<--(k:AngerComment)
                    WHERE c.uid = '{cluster.uid}' and k.embedding IS NOT NULL
                    RETURN k
                    """
    )
    return __extract_results(results, AngerComment)


def __get_joy_comments_for_given_cluster(cluster: Cluster) -> List[JoyComment]:
    results, _ = db.cypher_query(
        f"""
                        MATCH (c:Cluster)<--(k:JoyComment)
                        WHERE c.uid = '{cluster.uid}' and k.embedding IS NOT NULL
                        RETURN k
                        """
    )
    return __extract_results(results, JoyComment)


def __get_surprise_comments_for_given_cluster(cluster: Cluster) -> List[SurpriseComment]:
    results, _ = db.cypher_query(
        f"""
                            MATCH (c:Cluster)<--(k:SurpriseComment)
                            WHERE c.uid = '{cluster.uid}' and k.embedding IS NOT NULL
                            RETURN k
                            """
    )
    return __extract_results(results, SurpriseComment)


def __get_sadness_comments_for_given_cluster(cluster: Cluster) -> List[SadnessComment]:
    results, _ = db.cypher_query(
        f"""
                            MATCH (c:Cluster)<--(k:SadnessComment)
                            WHERE c.uid = '{cluster.uid}' and k.embedding IS NOT NULL
                            RETURN k
                            """
    )
    return __extract_results(results, SadnessComment)


def get_number_of_topics():
    results, _ = db.cypher_query(
        """
        MATCH (t:Topic)
        RETURN COUNT(t)
        """
    )

    return results


def get_number_of_article():
    results, _ = db.cypher_query(
        """
        MATCH (a:Article)
        RETURN COUNT(a)
        """
    )

    return results


def get_number_of_cluster():
    results, _ = db.cypher_query(
        """
        MATCH (c: Cluster)
        RETURN COUNT(c)
        """
    )

    return results


def get_number_of_comments():
    results, _ = db.cypher_query(
        """
        MATCH (c: Comment)
        RETURN COUNT(c)
        """
    )

    return results


def get_number_of_article_without_cluster():
    results, _ = db.cypher_query(
        """
        MATCH (a:Article)
        WHERE NOT (a)<--(:Cluster)
        RETURN a.article_id
        """
    )

    return results


def get_number_of_comments_without_cluster():
    results, _ = db.cypher_query(
        """
        MATCH (c:Comment)
        WHERE NOT (c)-->(:Cluster)
        RETURN COUNT(c)
        """
    )

    return results


def get_number_of_cluster_without_article():
    results, _ = db.cypher_query(
        """
        MATCH (c:Cluster)
        WHERE NOT (c) -->(:Article)
        RETURN COUNT(c)
        """
    )

    return results


def check_if_comment_with_id_is_in_db():
    results, _ = db.cypher_query(
        """
        MATCH (c:Comment)
        WHERE c.comment_id='3b498bc69f3446ce824a63953cdd0f54'
        RETURN c.text
        """
    )

    return results


def create_positive_query(clusters):
    where_clause = ""
    for cluster in clusters[:-1]:
        where_clause += f"""k.uid = '{cluster.uid}' or """

    where_clause += f"""k.uid = '{clusters[-1].uid}' """

    return f"""
                MATCH (k:Cluster)<--(c:PositiveComment)
                WHERE {where_clause}
                RETURN c
                """


def create_neutral_query(clusters):
    where_clause = ""
    for cluster in clusters[:-1]:
        where_clause += f"""k.uid = '{cluster.uid}' or """

    where_clause += f"""k.uid = '{clusters[-1].uid}' """

    return f"""
                MATCH (k:Cluster)<--(c:NeutralComment)
                WHERE {where_clause}
                RETURN c
                """


def create_negative_query(clusters):
    where_clause = ""
    for cluster in clusters[:-1]:
        where_clause += f"""k.uid = '{cluster.uid}' or """

    where_clause += f"""k.uid = '{clusters[-1].uid}' """

    return f"""
                MATCH (k:Cluster)<--(c:NegativeComment)
                WHERE {where_clause}
                RETURN c
                """


def get_sentiment_comments_for_given_clusters(clusters: List[Cluster]) -> tuple[
    list[PositiveComment], list[NeutralComment], list[NegativeComment]]:
    results_positive_comments = []
    results_neutral_comments = []
    results_negative_comments = []

    positive_query = create_positive_query(clusters)
    neutral_query = create_neutral_query(clusters)
    negative_query = create_negative_query(clusters)

    with driver.session(database='neo4j') as session:
        session.execute_read(process_cluster, positive_query, PositiveComment, results_positive_comments)
        session.execute_read(process_cluster, neutral_query, PositiveComment, results_neutral_comments)
        session.execute_read(process_cluster, negative_query, PositiveComment, results_negative_comments)

    return results_positive_comments, results_neutral_comments, results_negative_comments


def create_query(clusters, comment_type: str):
    where_clause = ""
    for cluster in clusters[:-1]:
        where_clause += f"""k.uid = '{cluster.uid}' or """

    where_clause += f"""k.uid = '{clusters[-1].uid}' """

    return f"""
                        MATCH (k:Cluster)<--(c:{comment_type})
                        WHERE {where_clause}
                        RETURN c
                        """


def get_emotion_comments_for_given_clusters(clusters: List[Cluster]) -> tuple[
    list[LoveComment], list[FearComment], list[AngerComment], list[JoyComment], list[SurpriseComment], list[
        SadnessComment]]:
    results_love_comments = []
    results_fear_comments = []
    results_anger_comments = []
    results_joy_comments = []
    results_surprise_comments = []
    results_sadness_comments = []

    love_query = create_query(clusters, "LoveComment")
    fear_query = create_query(clusters, "FearComment")
    anger_query = create_query(clusters, "AngerComment")
    joy_query = create_query(clusters, "JoyComment")
    surprise_query = create_query(clusters, "SurpriseComment")
    sadness_query = create_query(clusters, "SadnessComment")

    with driver.session(database='neo4j') as session:
        session.execute_read(process_cluster, love_query, LoveComment, results_love_comments)
        session.execute_read(process_cluster, fear_query, FearComment, results_fear_comments)
        session.execute_read(process_cluster, anger_query, AngerComment, results_anger_comments)
        session.execute_read(process_cluster, joy_query, JoyComment, results_joy_comments)
        session.execute_read(process_cluster, surprise_query, SurpriseComment, results_surprise_comments)
        session.execute_read(process_cluster, sadness_query, SadnessComment, results_sadness_comments)

    return (results_love_comments, results_fear_comments, results_anger_comments, results_joy_comments,
            results_surprise_comments, results_sadness_comments)


def get_news_agency_comments_for_given_cluster(ny_times_cluster, breitbart_cluster) -> tuple[
    list[Comment], list[Comment]]:
    results_ny_times_comments = []
    results_breitbart_comments = []

    start = time.time()

    # Use Pro and ContraComment labels to get all comment for specific news agency.
    # Using Comment label would result in getting same comments for all labels increasing retrieval time

    pro_ny_times_query = create_query(ny_times_cluster, "ProComment")
    contra_ny_times_query = create_query(ny_times_cluster, "ContraComment")
    pro_breitbart_query = create_query(breitbart_cluster, "ProComment")
    contra_breitbart_query = create_query(breitbart_cluster, "ContraComment")

    print(f"Created queries {time.time() - start}")

    with driver.session(database="neo4j") as session:
        session.execute_read(process_cluster, pro_ny_times_query, Comment, results_ny_times_comments)
        session.execute_read(process_cluster, contra_ny_times_query, Comment, results_ny_times_comments)
        session.execute_read(process_cluster, pro_breitbart_query, Comment, results_breitbart_comments)
        session.execute_read(process_cluster, contra_breitbart_query, Comment, results_breitbart_comments)


    print(f"Executed queries {time.time() - start}")

    ny_times_comments_seen = set()
    ny_times_comments = [comment for comment in results_ny_times_comments if comment.text not in ny_times_comments_seen and not
    ny_times_comments_seen.add(comment.text)]

    print(f"Removed duplicates from results ny_times_comments_seen: {time.time() - start}")

    breitbart_comments_seen = set()
    breitbart_comments = [comment for comment in results_breitbart_comments if comment.text not in breitbart_comments_seen and
                          not breitbart_comments_seen.add(comment.text)]


    print(f"Removed duplicates from results breitbart_comments_seen: {time.time() - start}")

    return ny_times_comments, breitbart_comments


def delete_article_nodes_without_cluster():
    results, _ = db.cypher_query(f"""
        MATCH (a:Article)
        WHERE NOT (a)<--(:Cluster)
        DETACH DELETE a
        """)


def delete_cluster_without_article():
    results, _ = db.cypher_query(
        """
        MATCH (c:Cluster)
        WHERE NOT (c) -->(:Article)
        DETACH DELETE c
        """
    )


def delete_comments_without_article():
    results, _ = db.cypher_query(
        """
        MATCH (c:Comment)
        WHERE NOT (c) -->(:Cluster)
        DETACH DELETE c
        """
    )
