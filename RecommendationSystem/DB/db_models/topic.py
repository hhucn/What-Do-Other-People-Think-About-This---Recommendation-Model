from neomodel import StructuredNode, UniqueIdProperty, StringProperty, RelationshipFrom, ArrayProperty


class Topic(StructuredNode):
    topic_id = UniqueIdProperty()
    topic_name = StringProperty()
    embedding = ArrayProperty()

    article = RelationshipFrom('.article.Article', 'BELONGS_TO')
