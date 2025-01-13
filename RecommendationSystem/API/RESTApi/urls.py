from django.urls import path

from . import views

urlpatterns = [
    path('stance', views.get_stance_recommendations, name='get-stance-recommendations'),
    path('sentiment', views.get_sentiment_recommendations, name='get-sentiment-recommendations'),
    path('emotion', views.get_emotion_recommendations, name='get-emotion-recommendations'),
    path('news-agency', views.get_news_agency_recommendations, name='get-news-agency-recommendations'),
    path('random', views.get_random_recommendations, name="get-random-recommendations"),
    path('statistics', views.get_db_statistics, name='get-db-statistics'),
    path('broken-nodes', views.get_number_of_broken_nodes, name='broken-nodes'),
    path('delete-broken-nodes', views.delete_broken_nodes, name='delete-broken-nodes'),
    path('similarity', views.get_similarity_recommendations, name='get-similarity-recommendations'),
    path('multi', views.get_multiprocessed_data, name='get-multiprocessed-data'),

]
