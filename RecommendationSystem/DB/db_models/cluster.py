from neomodel import StructuredNode, ArrayProperty, RelationshipTo, RelationshipFrom, StringProperty, UniqueIdProperty


class Cluster(StructuredNode):
    uid = UniqueIdProperty()
    cluster_name = StringProperty()
    most_frequent_words = ArrayProperty()
    embedding = ArrayProperty()

    article = RelationshipTo('.article.Article', 'BELONGS_TO')
    comment = RelationshipFrom('.comment.Comment', 'BELONGS_TO')
