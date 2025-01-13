import logging
from multiprocessing import Pool
from random import random
from time import sleep

from django.http import JsonResponse, HttpRequest

from RecommendationSystem.DB.utils import get_number_of_topics, get_number_of_article, get_number_of_cluster, get_number_of_comments, get_number_of_article_without_cluster, get_number_of_comments_without_cluster, get_number_of_cluster_without_article, check_if_comment_with_id_is_in_db, \
    delete_article_nodes_without_cluster, delete_cluster_without_article, delete_comments_without_article
from RecommendationSystem.Embedder.embedding_model import EmbeddingModel
from RecommendationSystem.Model.EmotionRecommendationModel import EmotionRecommendationModel
from RecommendationSystem.Model.NewsAgencyRecommendationModel import NewsAgencyRecommendationModel
from RecommendationSystem.Model.RandomRecommendationModel import RandomRecommendationModel
from RecommendationSystem.Model.SentimentRecommendationModel import SentimentRecommendationModel
from RecommendationSystem.Model.SimilarityRecommendationModel import SimilarityRecommendationModel
from RecommendationSystem.Model.StanceRecommendationModel import StanceRecommendationModel
from RecommendationSystem.SemanticSimilarity.SemanticSimilarity import SemanticSimilarity

similarity_model = SemanticSimilarity()
embedding_model = EmbeddingModel()

stance_model = StanceRecommendationModel(similarity_model, embedding_model)
sentiment_model = SentimentRecommendationModel(similarity_model, embedding_model)
emotion_model = EmotionRecommendationModel(similarity_model, embedding_model)
news_agency_model = NewsAgencyRecommendationModel(similarity_model, embedding_model)
random_model = RandomRecommendationModel()
similarity_recommendation_model = SimilarityRecommendationModel(similarity_model=similarity_model, embedding_model=embedding_model)


def get_random_recommendations(request: HttpRequest) -> JsonResponse:
    article_id = request.GET.get("article_id")
    logging.info("Get Random Recommendations")
    random_comments = random_model.get_recommendations(article_id)

    return JsonResponse({"Random-Comments": random_comments})


def get_similarity_recommendations(request: HttpRequest) -> JsonResponse:
    print("Get Similarity Recommendations")
    article_id = request.GET.get("article_id")
    user_comment = request.GET.get("user_comment")
    print(f"Got article id {article_id} and user_comment {user_comment}")

    similarity_recommendations = similarity_recommendation_model.get_recommendations(article_id, user_comment)

    print(similarity_recommendations)

    return JsonResponse({"Similarity-Recommendations": similarity_recommendations})


def get_stance_recommendations(request: HttpRequest) -> JsonResponse:
    """
    Receives Http request, extracts user comment and other information and triggers model to get suitable recommendations.
    Sends the recommendations as a JSON response back to user interface.
    :param request: request where the data for the recommendation model are extracted.
    :return: Json Response with comment recommendations.
    """

    # Retrieve here the information needed by the model from the request
    comment_data = extract_comment_data(request)

    if len(comment_data.keys()) == 0:
        return []

    # Replace with actual model that inherits form the abstract superclass
    logging.info("Get Stance Recommendations")
    relevant_stance_comments = stance_model.get_recommendations(comment_data)

    return JsonResponse({"Stance-Comments": relevant_stance_comments})


def get_sentiment_recommendations(request: HttpRequest) -> JsonResponse:
    """
    Receives Http request, extracts user comment and other information and triggers model to get suitable recommendations.
    Sends the recommendations as a JSON response back to user interface.
    :param request: request where the data for the recommendation model are extracted.
    :return: Json Response with comment recommendations.
    """

    # Retrieve here the information needed by the model from the request
    comment_data = extract_comment_data(request)

    if len(comment_data.keys()) == 0:
        return []

    # Replace with actual model that inherits form the abstract superclass
    logging.info("Get Sentiment Recommendations")
    relevant_comments = sentiment_model.get_recommendations(comment_data)

    return JsonResponse({"Sentiment-Comments": relevant_comments})


def get_emotion_recommendations(request: HttpRequest) -> JsonResponse:
    """
    Receives Http request, extracts user comment and other information and triggers model to get suitable recommendations.
    Sends the recommendations as a JSON response back to user interface.
    :param request: request where the data for the recommendation model are extracted.
    :return: Json Response with comment recommendations.
    """

    # Retrieve here the information needed by the model from the request
    comment_data = extract_comment_data(request)

    if len(comment_data.keys()) == 0:
        return []

    # Replace with actual model that inherits form the abstract superclass
    logging.info("Get emotion Recommendations")
    relevant_comments = emotion_model.get_recommendations(comment_data)

    return JsonResponse({"Emotion-Comments": relevant_comments})


def do_stuff(start):
    return [i * i for i in range(start, 10000)]


def get_multiprocessed_data(request: HttpRequest) -> JsonResponse:
    with Pool(processes=2) as pool:
        results = []

        i = 0

        foo = pool.apply(do_stuff, args=(i,))

        results.extend(foo)

    return JsonResponse({
        "results": results
    })


def get_news_agency_recommendations(request: HttpRequest) -> JsonResponse:
    """
    Receives Http request, extracts user comment and other information and triggers model to get suitable recommendations.
    Sends the recommendations as a JSON response back to user interface.
    :param request: request where the data for the recommendation model are extracted.
    :return: Json Response with comment recommendations.
    """

    # Retrieve here the information needed by the model from the request
    comment_data = extract_comment_data(request)

    if len(comment_data.keys()) == 0:
        return []

    # Replace with actual model that inherits form the abstract superclass
    logging.info("Get News Agency Recommendations")
    relevant_comments = news_agency_model.get_recommendations(comment_data)

    return JsonResponse({"News-Agency-Comments": relevant_comments})


def extract_comment_data(request):
    comment_data: dict = {
        "user_comment": request.GET.get("user_comment"),
        "keywords": request.GET.get("keywords")
    }
    return comment_data


def get_db_statistics(request: HttpRequest) -> JsonResponse:
    return JsonResponse({
        "Number of Topics": get_number_of_topics(),
        "Number of Article": get_number_of_article(),
        "Number of Cluster": get_number_of_cluster(),
        "Number of Comments": get_number_of_comments()
    })


def get_number_of_broken_nodes(request: HttpRequest) -> JsonResponse:
    return JsonResponse({
        "Number of Article": get_number_of_article_without_cluster(),
        "Number of Comments:": get_number_of_comments_without_cluster(),
        "Number Of Cluster:": get_number_of_cluster_without_article(),
        "Comment": check_if_comment_with_id_is_in_db()
    })


def delete_broken_nodes(request):
    delete_cluster_without_article()
    delete_article_nodes_without_cluster()
    delete_comments_without_article()
