import datetime
import logging
from typing import List

from scrapy import Selector
from scrapy.loader import ItemLoader

from RecommendationSystem.NewsAgencyScraper.NewsAgencyScraper.items import ArticleDataItem

from RecommendationSystem.NewsAgencyScraper.NewsAgencyScraper.spiders.NewsAgenciesSpyder import \
    NewsAgenciesSpyder

from RecommendationSystem.NewsAgencyScraper.NewsAgencyScraper.spiders.spyder_utils import parse_pub_date


class NewsAgencySpyder(NewsAgenciesSpyder):
    name = "NewsAgencySpyder"
    
    def __init__(self):
        self.news_agency = "<news_agency>"
        self.news_agency_url = "<www.news_agency.com>"
        self.article_start_page_xpath = "<xpath>"
        self.is_relative_urls = False

    def extract_article_data(self, article_data_item: ArticleDataItem, article_selector: Selector) -> ItemLoader:
        logging.info("Parse article response")

        loader: ItemLoader = ItemLoader(item=article_data_item, selector=article_selector)

        loader.add_xpath("article_title", "<xpath_to_article_title>")

        # Parse keywords from url
        loader.add_value("keywords", self.__parse_keywords(article_data_item["url"][0]))

        pub_date = article_selector.xpath("<xpath_to_pub_date").get()
        loader.add_value("pub_date", parse_pub_date(pub_date))

        return loader

    def extract_comment_section_data(self, article_data_item, comment_selector: Selector) -> List:
        loader: ItemLoader = ItemLoader(item=article_data_item, selector=comment_selector)

        #driver = get_selenium_chrome_driver()
        #driver.get(article_data_item["url"][0])
        #selector = Selector(text=driver.page_source)

        comments = []
        # Add comments to list and store them in ItemLoader

        loader.add_value('comments', comments)

        return loader

    @staticmethod
    def __parse_keywords(url):
        url_parts = url.split("/")
        print(url_parts)
        return url_parts[-2].replace("-", " ")
