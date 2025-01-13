from typing import List

import gensim
import spacy

from RecommendationSystem.Clustering.TopicClustering.gsdmm import MovieGroupProcess
from RecommendationSystem.DB.db_models.article import Article
from RecommendationSystem.DB.db_models.cluster import Cluster
from RecommendationSystem.DB.db_models.comment import Comment
from RecommendationSystem.Embedder.embedding_model import EmbeddingModel


class Clustering:
    def __init__(self, embedding_model: EmbeddingModel):
        self.__number_of_clusters = 10
        self.__alpha = 0.01
        self.__beta = 0.01
        self.__n_iters = 30
        self.clustering_model = MovieGroupProcess(K=self.__number_of_clusters, alpha=self.__alpha
                                                  , beta=self.__beta, n_iters=self.__n_iters)
        self.embedding_model = embedding_model
        self.nlp = spacy.load('en_core_web_sm', disable=['parser', 'ner'])

    def create_cluster_nodes_for_article_with_comments(self, article: Article, comments: List[Comment]) -> None:
        if article is None or len(comments) == 0:
            return
        preprocessed_comments = self.train_topic_cluster_model(comments)
        cluster_list = self.create_cluster_nodes(article)

        for i, comment in enumerate(comments):
            cluster_number_for_comment = self.clustering_model.choose_best_label(preprocessed_comments[i])[0]
            article.cluster.connect(cluster_list[cluster_number_for_comment])
            cluster_list[cluster_number_for_comment].comment.connect(comment)

    def train_topic_cluster_model(self, comments: List[Comment]) -> List:
        preprocessed_comments = self.preprocess_comments_for_clustering([comment.text for comment in comments])
        topic_cluster_vocab = set(x for comment in preprocessed_comments for x in comment)
        topic_cluster_n_terms = len(topic_cluster_vocab)
        self.clustering_model.fit(preprocessed_comments, topic_cluster_n_terms)
        return preprocessed_comments

    def create_cluster_nodes(self, article):
        cluster_list = []
        for cluster_number in range(self.__number_of_clusters):
            cluster_top_n_words = self.extract_top_n_words_for_cluster(cluster_number)
            embedding = None
            if len(cluster_top_n_words) != 0:
                embedding = self.embedding_model.embed_vector(cluster_top_n_words).tolist()
            cluster: Cluster = Cluster(
                cluster_name=article.article_id + "cluster_number " + str(cluster_number),
                most_frequent_words=cluster_top_n_words,
                embedding= embedding
            ).save()
            cluster_list.append(cluster)
        return cluster_list

    def extract_top_n_words_for_cluster(self, cluster_number):
        most_frequent_words_sorted = sorted(self.clustering_model.cluster_word_distribution[cluster_number].items(),
                                            key=lambda k: k[1], reverse=True)[:10]
        cluster_top_n_words = [x[0] for x in most_frequent_words_sorted]
        return cluster_top_n_words

    def preprocess_comments_for_clustering(self, comments: List[str]):
        tokenized_comments = list(self.sentence_to_words(comments))
        n_grams = self.make_n_grams(tokenized_comments)
        lemmatized_comments = self.lemmatize_comments(n_grams, allowed_postags=['NOUN', 'ADJ'])
        return lemmatized_comments

    def sentence_to_words(self, comments):
        for comment in comments:
            yield (gensim.utils.simple_preprocess(str(comment), deacc=True))

    def make_n_grams(self, texts):
        bigram = gensim.models.Phrases(texts, min_count=5, threshold=100)  # higher threshold fewer phrases.
        bigram_mod = gensim.models.phrases.Phraser(bigram)
        trigram = gensim.models.Phrases(bigram[texts], threshold=100)
        trigram_mod = gensim.models.phrases.Phraser(trigram)
        bigrams_text = [bigram_mod[doc] for doc in texts]
        trigrams_text = [trigram_mod[bigram_mod[doc]] for doc in bigrams_text]
        return trigrams_text

    def lemmatize_comments(self, texts, allowed_postags=['NOUN', 'ADJ']):
        texts_out = []
        for sent in texts:
            doc = self.nlp(" ".join(sent))
            texts_out.append([token.lemma_ for token in doc if token.pos_ in allowed_postags])
        return texts_out
