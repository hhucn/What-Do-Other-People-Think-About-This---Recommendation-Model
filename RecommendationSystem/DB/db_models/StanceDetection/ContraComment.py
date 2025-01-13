from neomodel import StructuredNode, StringProperty, ArrayProperty, RelationshipTo, UniqueIdProperty

from RecommendationSystem.DB.db_models.comment import Comment


class ContraComment(Comment):
    """
    Neomodel structured node that defines the properties for the comments that are stored in the Neo4J database
    """

