from neomodel import StructuredNode, StringProperty, ArrayProperty, RelationshipFrom, UniqueIdProperty, DateProperty, \
    RelationshipTo


class Article(StructuredNode):
    """
    Neomodel structured node that defines the properties for the articles that are stored in the Neo4J database
    """
    article_id = UniqueIdProperty()
    article_title = StringProperty()
    news_agency = StringProperty()
    keywords = StringProperty()
    pub_date = DateProperty()
    embedding = ArrayProperty()
    url = StringProperty()

    cluster = RelationshipFrom('.cluster.Cluster', 'BELONGS_TO')
    topic = RelationshipTo('.topic.Topic', 'BELONGS_TO')

    def __eq__(self, other):
        if self.article_id == other.article_id:
            return True
        return False

    def __ne__(self, other):
        if self.article_id != other.article_id:
            return True
        return False

    def __hash__(self):
        return hash(self.article_id)
