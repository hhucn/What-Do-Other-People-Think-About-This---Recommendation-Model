import os
import sys
from typing import List

from neomodel import config
from tqdm import tqdm

from RecommendationSystem.DB.db_models.article import Article
from RecommendationSystem.DB.db_models.comment import Comment
from RecommendationSystem.Embedder.embedding_model import EmbeddingModel
from RecommendationSystem.DB.utils import get_articles_without_embedding, \
    get_comments_without_embedding, update_node_with_embedding


config.DATABASE_URL = os.environ.get("NEO4J_BOLT_URL", 'bolt://neo4j:test@neo4j:7687')


def main() -> None:
    """
    Queries all article and comment nodes from the Neo4J database, calls the embedding method for properties of the
    nodes that should be embedded and then updates the nodes in the database with the embeddings.
    """
    embedder = EmbeddingModel()

    # Query all article and comments from the database without an embedding
    articles: List[Article] = get_articles_without_embedding()
    comments: List[Comment] = get_comments_without_embedding()

    for article in tqdm(articles):
        article_embedding = embedder.embed("<article field you want to embed>")
        update_node_with_embedding(article, article_embedding)

    for comment in tqdm(comments):
        comment_embedding = embedder.embed(comment.text)
        update_node_with_embedding(comment, comment_embedding)


if __name__ == '__main__':
    main()
