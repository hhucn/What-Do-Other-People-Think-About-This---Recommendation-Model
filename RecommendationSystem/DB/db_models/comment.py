from neomodel import StructuredNode, StringProperty, ArrayProperty, RelationshipTo, UniqueIdProperty, FloatProperty


class Comment(StructuredNode):
    """
    Neomodel structured node that defines the properties for the comments that are stored in the Neo4J database
    """
    comment_id = UniqueIdProperty()
    text = StringProperty()
    embedding = ArrayProperty()
    stances = ArrayProperty()
    relevance_score = FloatProperty()
    reason_score = FloatProperty()
    source_score = FloatProperty()
    example_score = FloatProperty()
    personal_story_score = FloatProperty()

    article = RelationshipTo('.cluster.Cluster', 'BELONGS_TO')
