import logging
import os
import sys
import time
from datetime import datetime
from typing import List

from selenium import webdriver
from selenium.common import NoSuchElementException, StaleElementReferenceException, ElementClickInterceptedException, \
    TimeoutException
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.chrome.webdriver import WebDriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.wait import WebDriverWait
from sentence_transformers import SentenceTransformer, util
from webdriver_manager.chrome import ChromeDriverManager
from selenium.webdriver.support import expected_conditions as EC

def parse_pub_date(pub_date: str) -> datetime.date:
    if pub_date is None:
        return datetime.now().date()
    date = pub_date.split("T")[0]
    year = int(date.split("-")[0])
    month = int(date.split("-")[1])
    day = int(date.split("-")[2])
    return datetime(year=year, month=month, day=day).date()

def click_button_to_load_all_comments(driver: WebDriver, button_xpath: str) -> None:
    """
    Clicks the load new comments button to dynamically load all comments until all comments are loaded
    :param driver: Webdriver that queries the website
    :param button_xpath: XPath to button that should be clicked
    :return:
    """
    try:
        counter = 0
        while driver.find_element(By.XPATH, button_xpath) and counter < 1000:
            WebDriverWait(driver, 40).until(EC.element_to_be_clickable((By.XPATH, button_xpath)))
            driver.find_element(By.XPATH, button_xpath).click()
            time.sleep(10)
            logging.info("Loaded more comments")
            counter = counter + 1
    except (NoSuchElementException, StaleElementReferenceException, ElementClickInterceptedException, TimeoutException):
        logging.info("Cannot find Load more comments button")
