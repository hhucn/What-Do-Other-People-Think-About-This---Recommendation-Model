import datetime
import logging
import os
import time
from csv import DictReader
from typing import List

import scrapy
from scrapy import Selector
from scrapy.http import Response
from scrapy.loader import ItemLoader
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


class Breitbart(scrapy.Spider):
    name = "Breitbart"

    def __init__(self):
        self.news_agency = "Breitbart"
        self.driver = self.get_firefox_driver()

    def start_requests(self):
        logging.info("Scrape Breitbart Article")

        with open("RecommendationSystem/static/Scraper/Breitbart/Breitbart_Article_World_URLs.csv", encoding="utf-8-sig") as csv_file:
            csv_reader = DictReader(csv_file, delimiter=";")

            for article in csv_reader:
                url = article["webURL"]
                logging.info(url)
                comment_section_url = article["commentSectionURL"]
                yield scrapy.Request(url=url, callback=self.parse, meta={"comment_section_url": comment_section_url})

    def parse(self, response: Response, **kwargs):
        """
        Parses the start page of the news agency
        :param response:
        :param kwargs:
        :return:
        """
        logging.info("Parse start page of " + self.news_agency)

        loader: ItemLoader = ItemLoader(item=ArticleDataItem(), response=response)

        loader.add_value('url', response.url)

        loader.add_value('news_agency', self.news_agency)

        loader.add_xpath("article_title", "//h1/text()")

        # Parse keywords from url
        loader.add_xpath("keywords", "//meta[@name='news_keywords']/@content")

        pub_date = response.xpath("//meta[@property='article:published_time']/@content").get()
        loader.add_value("pub_date", parse_pub_date(pub_date))

        self.driver.get(response.meta["comment_section_url"])

        WebDriverWait(self.driver, 20).until(EC.presence_of_element_located((By.XPATH, "//div[@class='post-message ']")))

        click_button_to_load_all_comments(self.driver, '//a[contains(@class, "load-more-refresh__button")]')

        selector = Selector(text=self.driver.page_source)

        comments = selector.xpath("//div[contains(@class,'post-message')]/div/p/text()").getall()
        # Add comments to list and store them in ItemLoader
        loader.add_value('comments', comments)

        yield loader.load_item()

    def get_firefox_driver(self):
        opts = FirefoxOptions()
        opts.add_argument("--headless")
        chrome_logger = logging.getLogger('selenium.webdriver.remote.remote_connection')
        chrome_logger.setLevel(logging.INFO)
        driver = webdriver.Firefox(service=FirefoxService(GeckoDriverManager().install()), options=opts)
        return driver

    @staticmethod
    def __parse_keywords(url):
        url_parts = url.split("/")
        print(url_parts)
        return url_parts[-2].replace("-", " ")
