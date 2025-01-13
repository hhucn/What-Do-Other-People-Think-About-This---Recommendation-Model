import datetime
import logging
from typing import List

from scrapy import Selector
from scrapy.loader import ItemLoader

from RecommendationSystem.NewsAgencyScraper.NewsAgencyScraper.items import ArticleDataItem

from RecommendationSystem.NewsAgencyScraper.NewsAgencyScraper.spiders.NewsAgenciesSpyder import \
    NewsAgenciesSpyder

from RecommendationSystem.NewsAgencyScraper.NewsAgencyScraper.spiders.spyder_utils import parse_pub_date,\
    click_button_to_load_all_comments



class NYTimesSpyder(NewsAgenciesSpyder):
    name = "NYTimesSpyder"
    
    def __init__(self):
        self.news_agency = "NewYorkTimes"
        self.news_agency_url = "https://www.nytimes.com/"
        self.article_start_page_xpath = "//section[@class='story-wrapper']/a/@href"
        self.is_relative_urls = False

    def extract_article_data(self, article_data_item: ArticleDataItem, article_selector: Selector) -> ItemLoader:
        logging.info("Parse article response")

        loader: ItemLoader = ItemLoader(item=article_data_item, selector=article_selector)

        loader.add_xpath("article_title", "//meta[@property='og:title']/@content")

        loader.add_xpath("keywords", "//meta[@name='news_keywords']/@content")

        pub_date = article_selector.xpath("//meta[@property='article:published_time']/@content").get()
        loader.add_value("pub_date", parse_pub_date(pub_date))

        print(loader)

        return loader

    def extract_comment_section_data(self, article_data_item, comment_selector: Selector) -> List:
        loader: ItemLoader = ItemLoader(item=article_data_item, selector=comment_selector)

        driver = get_selenium_chrome_driver()
        driver.get(article_data_item["url"][0] + '#commentsContainer')

        click_button_to_load_all_comments(driver, '//button[contains(@class, "css-1rifrtd")]')

        selector = Selector(text=driver.page_source)

        comments = []
        # Add comments to list and store them in ItemLoader
        comments = selector.xpath("//p[contains(@id, 'comment-content')]/text()").getall()

        loader.add_value('comments', comments)

        return loader

    @staticmethod
    def __parse_keywords(url):
        url_parts = url.split("/")
        print(url_parts)
        return url_parts[-2].replace("-", " ")
