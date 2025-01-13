import datetime
import logging
import os
import time
from typing import List

from scrapy import Selector
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


class Breitbart(NewsAgenciesSpyder):
    name = "Breitbart"

    def __init__(self):
        self.news_agency = "Breitbart"
        self.news_agency_base_url = "https://www.breitbart.com"
        self.article_start_page_xpath = "//article/a/@href"
        # self.news_agency_url = 'https://www.breitbart.com/tag/donald-trump/'
        # self.news_agency_url = 'https://www.breitbart.com/tag/abortion/'
        self.news_agency_url = 'https://www.breitbart.com/tag/climate-change/'
        # self.article_urls = [
        #                    #"https://www.breitbart.com/politics/2017/01/11/trump-on-the-us-mexico-border-wall-were-going-to-start-building/",
        #                    #"https://www.breitbart.com/local/2017/06/25/trumps-border-wall-construction-already-behind-schedule/",
        #                    #"https://www.breitbart.com/politics/2017/07/10/exclusive-mark-meadows-draws-line-in-the-sand-next-spending-bill-must-fund-president-trumps-border-wall/",
        #                    #"https://www.breitbart.com/politics/2018/11/26/illegal-immigration-under-trump-on-track-to-hit-highest-level-in-a-decade/",
        #                    #"https://www.breitbart.com/border/2018/11/22/donald-trump-says-may-close-us-mexico-border/",
        #                    #"https://www.breitbart.com/politics/2018/09/08/president-trump-may-use-military-to-build-border-wall/"
        #                    #"https://www.breitbart.com/politics/2018/07/18/house-funds-200-miles-border-barrier-2019/",
        #                    #"https://www.breitbart.com/border/2019/07/23/exclusive-completed-section-of-trumps-new-wall-helping-secure-border-say-agents/",
        #                    #"https://www.breitbart.com/politics/2019/05/13/poll-gop-voters-say-border-wall-reducing-all-immigration-must-be-top-priority-for-trump/",
        #                    #"https://www.breitbart.com/politics/2019/01/08/transcript-president-donald-trumps-oval-office-address-on-the-border-crisis/"
        #                    #"https://www.breitbart.com/politics/2018/12/04/conservative-leaders-demand-congress-fund-border-wall-national-security/",
        #                    #"https://www.breitbart.com/politics/2018/01/05/border-wall-cost-1-8-billion-annual-10-years/",
        #                    #"https://www.breitbart.com/politics/2018/03/25/despite-trumps-claims-1-6b-border-fencing/",
        #                    #"https://www.breitbart.com/politics/2018/04/10/trumps-new-border-wall-resembles-fence-constructed-under-obama-that-illegal-aliens-recently-hopped-over/"
        #                    "https://www.breitbart.com/politics/2017/02/27/full-transcript-president-donald-trumps-exclusive-interview-breitbart-news-network-oval-office/" ,
        #                    "https://www.breitbart.com/2020-election/2019/06/17/exclusive-pro-trump-pac-eyes-big-investment-in-2020-battlegrounds/",
        #                    "https://www.breitbart.com/clips/2019/06/19/trump-those-who-know-a-nation-must-care-for-its-own-make-up-our-movement/",
        #                    "https://www.breitbart.com/border/2017/01/20/trump-clamp-immigration-merit-based-system/",
        #                    ]
        self.is_relative_urls = True
        self.driver = self.get_firefox_driver()

    def extract_article_data(self, article_data_item: ArticleDataItem, article_selector: Selector) -> ItemLoader:
        logging.info("Parse article response")

        loader: ItemLoader = ItemLoader(item=article_data_item, selector=article_selector)

        loader.add_xpath("article_title", "//h1/text()")

        # Parse keywords from url
        loader.add_xpath("keywords", "//meta[@name='news_keywords']/@content")

        pub_date = article_selector.xpath("//meta[@property='article:published_time']/@content").get()
        loader.add_value("pub_date", parse_pub_date(pub_date))

        return loader

    def extract_comment_section_data(self, article_data_item, comment_selector: Selector) -> List:
        loader: ItemLoader = ItemLoader(item=article_data_item, selector=comment_selector)
        print("COMMENTS")

        comment_url = comment_selector.xpath("//a[@class='d-comments-button']/@href").get()
        print("COMMENTS_URL:")
        print(comment_url)
        self.driver.get(comment_url)

        try:
            WebDriverWait(self.driver, 20).until(EC.presence_of_element_located((By.XPATH, "//div[@class='disqus_thread']/iframe")))
        except TimeoutException:
            print("DISQUS THREAD NOT FOUND")
            time.sleep(10)

        selector = Selector(text=self.driver.page_source)
        comment_url_1 = selector.xpath("//div[@id='disqus_thread']").get()
        comment_url_2 = selector.xpath("//div[@id='disqus_thread']/iframe").get()
        comment_url_3 = selector.xpath("//div[@id='disqus_thread']/iframe/@src").get()
        comment_url_4 = selector.xpath("//div[@id='disqus_thread']/iframe[contains('@id', 'dsq')]").get()
        comment_url = selector.xpath("//div[@id='disqus_thread']/iframe/@src").get()
        print("COMMENTS_URL_IFRAME")
        print(comment_url_1)
        print(comment_url_2)
        print(comment_url_3)
        print(comment_url_4)
        print(comment_url)

        self.driver.get(comment_url)
        WebDriverWait(self.driver, 20).until(EC.presence_of_element_located((By.XPATH, "//div[@class='post-message ']")))

        click_button_to_load_all_comments(self.driver, '//a[contains(@class, "load-more-refresh__button")]')

        selector = Selector(text=self.driver.page_source)

        print(self.driver.page_source)

        comments = selector.xpath("//div[contains(@class,'post-message')]/div/p/text()").getall()
        print(comments)
        # Add comments to list and store them in ItemLoader
        loader.add_value('comments', comments)

        return loader

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
