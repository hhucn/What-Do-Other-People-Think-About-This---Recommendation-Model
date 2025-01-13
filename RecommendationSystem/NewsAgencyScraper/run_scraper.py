import os
import sys

from scrapy.crawler import CrawlerRunner
from scrapy.utils.log import configure_logging
from scrapy.utils.project import get_project_settings
from twisted.internet import reactor, defer

from NewsAgencyScraper.spiders.WashingtonTimesSpyder import WashingtonTimes
from NewsAgencyScraper.spiders.BreitbartCSVSpyder import Breitbart

# Source: https://shineosolutions.com/2018/09/13/running-a-web-crawler-in-a-docker-container/ accessed 02/03/2022

# Import your spyder like this
# from RecommendationSystem.NewsAgencyScraper.NewsAgencyScraper.spiders.<FileName> import <ClassName>

configure_logging()
settings_file_path = 'NewsAgencyScraper.settings'
os.environ.setdefault('SCRAPY_SETTINGS_MODULE', settings_file_path)
runner = CrawlerRunner(get_project_settings())

@defer.inlineCallbacks
def crawl() -> None:
    """
    Starts and organizes the different spiders to scrape the news agencies.
    :return: None
    """
    # Add one yield runner.crawl(<SpyderName>) per spyder
    yield runner.crawl(Breitbart)
    reactor.stop()


crawl()
reactor.run()
