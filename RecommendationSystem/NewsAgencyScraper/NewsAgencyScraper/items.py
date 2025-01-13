# Define here the models for your scraped items
#
# See documentation in:
# https://docs.scrapy.org/en/latest/topics/items.html
import re

import scrapy
from itemloaders.processors import MapCompose, Identity
from scrapy import Field


class NewsagencyscraperItem(scrapy.Item):
    # define the fields for your item here like:
    # name = scrapy.Field()
    pass


# Methods from https://www.kaggle.com/code/ptfrwrd/topic-modeling-guide-gsdm-lda-lsi/notebook
def remove_double_whitespace(comment):
    return re.sub('\s+', ' ', comment)


def remove_single_quote(comment):
    return re.sub("\'", '', comment)


def clean_comment(comment):
    comment = remove_double_whitespace(comment)
    comment = remove_single_quote(comment)
    if comment == " ":
        return ""
    return comment


class ArticleDataItem(scrapy.Item):
    """
    Scrapy item that defines the field of the data item where we store the article and comment data
    """
    article_title = Field()
    keywords = Field()
    news_agency = Field()
    pub_date = Field()
    url = Field()

    comments = Field(input_processor=MapCompose(clean_comment), output_processor=Identity())
