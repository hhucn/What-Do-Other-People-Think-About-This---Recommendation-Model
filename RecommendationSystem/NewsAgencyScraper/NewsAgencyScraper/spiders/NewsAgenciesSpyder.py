import logging
from typing import List

import scrapy
from scrapy import Selector
from scrapy.http import Response
from scrapy.loader import ItemLoader
from selenium import webdriver
from selenium.webdriver.chrome.options import Options

from RecommendationSystem.NewsAgencyScraper.NewsAgencyScraper.items import ArticleDataItem
from webdriver_manager.chrome import ChromeDriverManager




class NewsAgenciesSpyder(scrapy.Spider):
    def __init__(self, name=None, **kwargs):
        super().__init__(name, kwargs)
        self.article_urls = None

    def start_requests(self):
        """
        Starts the initial request on the news agency start page
        :return:
        """
        logging.info("Start scraping " + self.news_agency)

        yield scrapy.Request(url=self.news_agency_url, callback=self.parse)

    def parse(self, response, **kwargs):
        """
        Parses the start page of the news agency
        :param response:
        :param kwargs:
        :return:
        """
        logging.info("Parse start page of " + self.news_agency)

        articles_urls = response.xpath(self.article_start_page_xpath)

        for url_selector in articles_urls:
            loader: ItemLoader = ItemLoader(item=ArticleDataItem(), selector=url_selector)

            if self.is_relative_urls:
                loader.add_value('url', self.news_agency_base_url + url_selector.get())
            else:
                loader.add_value('url', url_selector.get())

            loader.add_value('news_agency', self.news_agency)

            article_data_item = loader.load_item()

            if self.news_agency_base_url in article_data_item["url"][0]:
                yield response.follow(url=article_data_item["url"][0], callback=self.parse_article,
                                 meta={"article_data_item": article_data_item})

    def parse_article(self, response):
        """
        Parses the article response to get the comment section
        :param response:
        :return:
        """
        logging.info("parse article response")

        article_selector = Selector(response=response, type='html')
        loader: ItemLoader = self.extract_article_data(response.meta["article_data_item"], article_selector)

        loader = self.extract_comment_section_data(loader.load_item(), article_selector)

        yield loader.load_item()


    @staticmethod
    def extract_article_data(article_data_item: ArticleDataItem, article_selector: Selector) -> ItemLoader:
        """
        Extracts all data from article and loads them into a ItemLoader
        :param article:
        :return:
        """
        loader: ItemLoader = ItemLoader(item=article_data_item, selector=article_selector)

        # Extract data from article selector and store them in loader
        # e.g. loader.add_xpath("article_title", "//title/text()")

        return loader

    @staticmethod
    def extract_comment_section_data(article_data_item, comment_selector: Selector) -> List:
        """
        Extracts all comments from the comment section and stores them into a Itemloader
        """
        # In case that the comment sections is loaded dynamically, we have to use Selenium
        # driver = get_selenium_chrome_driver()
        # driver.get(<comment_section_url>)
        # selector = Selector(text=driver.page_source)
        # loader: ItemLoader = ItemLoader(item=article_data_item, selector=comment_section_selector)

        loader: ItemLoader = ItemLoader(item=article_data_item, selector=comment_selector)

        # Extract data from article selector and store them in loader
        # comments = response.xpath("//div[contains(@class,'comment')]//p[@class='text']/text()").extract()
        # if comments is not None:
        #    loader.add_value("comments", comments)

        return loader
