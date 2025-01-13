import logging
import sys
from datetime import datetime
from typing import List

import scrapy
import spacy
from scrapy import Selector
from scrapy.loader import ItemLoader
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.chrome.service import Service
from sentence_transformers import SentenceTransformer
from selenium.webdriver.chrome.service import Service as ChromeService
from webdriver_manager.chrome import ChromeDriverManager
from webdriver_manager.firefox import GeckoDriverManager
from selenium.webdriver.firefox.service import Service as FirefoxService
from selenium.webdriver import FirefoxOptions

from RecommendationSystem.NewsAgencyScraper.NewsAgencyScraper.items import ArticleDataItem
from RecommendationSystem.NewsAgencyScraper.NewsAgencyScraper.spiders.spyder_utils import \
    click_button_to_load_all_comments


class WashingtonTimes(scrapy.Spider):
    name = "WashingtonTimes"

    def __init__(self):
        super(WashingtonTimes, self).__init__()
        self.embedding_model = SentenceTransformer('stsb-roberta-base-v2')
        self.nlp = spacy.load('en_core_web_sm', disable=['parser', 'ner'])

    def start_requests(self):
        """
        Starts the scraping of Daily Mail
        :return:
        """
        urls = [["https://www.washingtontimes.com/news/2023/jul/10/study-points-worse-mental-health-outcomes-women-wh/",
                 "https://comment.instiengage.com/live/comments/api/page?pageTitle=Study%20points%20to%20worse%20mental%20health%20outcomes%20for%20women%20who%20have%20abortions%20versus%20giving%20birth%20-%20Washington%20Times&enableNewAuthFlow=true&siteUUID=dff2a308-b1e2-451c-a49a-adadd8636788&integrationId=172bad30-6d59-40f0-b6d9-73707f6e99d8&extPageId=2834621-19b6bde&contentId=c7956309-7d5b-4d46-8ffe-c2ef8c2607ab&sessionUUID=1a5c58b7-0395-4eea-8205-4707f4b8f346&cookieId=1a5c58b7-0395-4eea-8205-4707f4b8f346&frameUuid=bca6e1c3-6263-4943-a6b2-b21183935f92&pageUrl=https%3A%2F%2Fwww.washingtontimes.com%2Fnews%2F2023%2Fjul%2F10%2Fstudy-points-worse-mental-health-outcomes-women-wh%2F"]]

        for article_url, comments_url in urls:
            yield scrapy.Request(url=article_url, callback=self.parse_article, meta={'comments_url': comments_url})

    def parse_article(self, response):
        logging.info("PARSE ARTICLE PAGE OF DAILY MAIL")

        loader = self.__extract_article_data(response)
        article_title = response.xpath("//meta[@property='og:title']/@content").get()
        keywords = self.__extract_keywords_from_title(article_title)
        loader.add_value('keywords', keywords)
        logging.info("PARSE ARTICLE RESPONSE")

        pub_date_dict = datetime.now()

        loader.add_value("pub_date", pub_date_dict)

        opts = FirefoxOptions()
        opts.add_argument("--headless")
        chrome_logger = logging.getLogger('selenium.webdriver.remote.remote_connection')
        chrome_logger.setLevel(logging.INFO)
        driver = webdriver.Firefox(service=FirefoxService(GeckoDriverManager().install()), options=opts)

        driver.get(response.meta['comments_url'])

        logging.info("GET COMMENT SECTION URL")
        comments = []
        click_button_to_load_all_comments(driver, '//button[contains(@class, "LoadMoreButton")]')

        scrapy_selector = Selector(text=driver.page_source)
        comment_selector = scrapy_selector.xpath(
            "//div[contains(@class, 'CommentsTree__ListWrapLoadStateContainer-sc-psk8fb-3 jNGwKD')]/div")
        comments = self.__extract_comments(comment_selector)

        loader.add_value('comments', value=comments)
        print("Comments added")
        yield loader.load_item()

    @staticmethod
    def __extract_comments(comments_selector: Selector) -> List:
        comments = []
        for comment_selector in comments_selector:
            comment_parts = comment_selector.xpath('div//p[contains(@class, "CommentText")]')
            comment = ""
            for part in comment_parts:
                comment_part = part.xpath('span/text()').get()
                comment = comment + "\n" + comment_part
            comments.append(comment)
        return comments

    @staticmethod
    def __extract_article_data(response) -> ItemLoader:
        """
        Extracts data from article
        :param article:
        :return:
        """
        loader = ItemLoader(item=ArticleDataItem(), selector=response)
        loader.add_xpath('article_title', xpath="//meta[@property='og:title']/@content")
        loader.add_xpath('url', xpath="//meta[@property='og:url']/@content")
        loader.add_value('news_agency', value="WashingtonTimes")
        return loader

    def __extract_keywords_from_title(self, title: str) -> List[str]:
        doc = self.nlp(title)
        return [token.lemma_ for token in doc if token.pos_ in ['NOUN']]
