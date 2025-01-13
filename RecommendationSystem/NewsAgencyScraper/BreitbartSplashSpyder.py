import logging
import os
import time
from typing import List

from scrapy import Selector
from scrapy.loader import ItemLoader
from scrapy_splash import SplashRequest
from selenium.common import TimeoutException

from RecommendationSystem.NewsAgencyScraper.NewsAgencyScraper.items import ArticleDataItem

from RecommendationSystem.NewsAgencyScraper.NewsAgencyScraper.spiders.NewsAgenciesSpyder import \
    NewsAgenciesSpyder

from RecommendationSystem.NewsAgencyScraper.NewsAgencyScraper.spiders.spyder_utils import parse_pub_date, \
    click_button_to_load_all_comments
from selenium import webdriver
from webdriver_manager.firefox import GeckoDriverManager
from selenium.webdriver.firefox.service import Service as FirefoxService
from selenium.webdriver import FirefoxOptions
from selenium.webdriver.support.wait import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.common.by import By

os.environ.get('GH_TOKEN')


class BreitbartSplash(NewsAgenciesSpyder):
    name = "BreitbartSplash"

    def __init__(self):
        self.news_agency = "Breitbart"
        self.news_agency_base_url = "https://www.breitbart.com"
        self.article_start_page_xpath = "//article/a/@href"
        # self.news_agency_url = 'https://www.breitbart.com/tag/donald-trump/'
        # self.news_agency_url = 'https://www.breitbart.com/tag/abortion/'
        self.news_agency_url = 'https://www.breitbart.com/tag/climate-change/'

        self.is_relative_urls = True

    def extract_article_data(self, article_data_item: ArticleDataItem, article_selector: Selector) -> ItemLoader:
        logging.info("Parse article response")

        loader: ItemLoader = ItemLoader(item=article_data_item, selector=article_selector)

        loader.add_xpath("article_title", "//h1/text()")

        # Parse keywords from url
        loader.add_xpath("keywords", "//meta[@name='news_keywords']/@content")

        pub_date = article_selector.xpath("//meta[@property='article:published_time']/@content").get()
        loader.add_value("pub_date", parse_pub_date(pub_date))

        return loader

    #def extract_comment_section_data(self, article_data_item: ArticleDataItem, comment_selector: Selector):
    #    loader: ItemLoader = ItemLoader(item=article_data_item, selector=comment_selector)
    #    logging.info("COMMENTS")
#
    #    comment_url = comment_selector.xpath("//a[@class='d-comments-button']/@href").get()
    #    print("COMMENTS_URL:")
    #    print(comment_url)
#
    #    yield SplashRequest(
    #        url=comment_url,
    #        callback=self.process_comments,
    #        args={
    #            'wait': 15
    #        },
    #        meta={
    #            'data_item': loader.load_item()
    #        }
    #    )
#
    #def process_comments(self, response):
    #    logging.info(response.xpath("//div[@id='disqus_thread']"))
    #    comments = response.xpath("//div[contains(@class,'post-message')]/div/p/text()").getall()
    #    print(comments)
    #    # Add comments to list and store them in ItemLoader
#
    #    data_item = response.request.meta['data_Item']
#
    #    loader = ItemLoader(item=data_item)
#
    #    loader.add_value('comments', comments)
#
    #    yield loader


    @staticmethod
    def __parse_keywords(url):
        url_parts = url.split("/")
        print(url_parts)
        return url_parts[-2].replace("-", " ")
