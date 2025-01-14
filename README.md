# What-Do-Other-People-Think-About-This---Recommendation-Model

This repository contains the implementation of our comment recommendation model for the paper `What Do Other people think about this? Recommending Relevant and diverse user comments in comment sections`. 

Our model primarily aims to suggest comments that correspond to the comment the user is interested in. It leverages an assortment of machine learning and natural language processing methodologies to accomplish two objectives: Initially, the goal is to pinpoint comments that not only offer strong reasoning supporting the comment's subject but also highlight comments presenting a range of perspectives on the matter.

The model was implemented using the framework available at https://pypi.org/project/comment-recommendation-framework/, which served as the foundation for our approach.



## Setup
Ensure that the following tools are installed:
* Docker
* Docker-Compose
* Python >= 3.10

## Environment Variables
The framework need some environment variables to be set for running properly. Please ensure that you have an ```.env```
file with the following variables:
* NEO4J_PASSWORD
* NEO4J_BOLT_URL (Format: `bolt://neo4j:<NEO4J_PASSWORD>@neo4j:7687`)


### Run different moduls with docker-compose
We provide you with the following `docker-compose` files to run the different components of the recommendation framework. 

* `docker-compose.scraping.yml`: Runs the news agency scraper to retrieve articles and comments from various news agencies.
* `docker-compose.test.yml`: Runs the tests for the system.
* `docker-compose.api.yml`: Runs the comment-recommendation systems.
* `docker-compose.csv.yml': Import datasets via csv file into the database. Make sure to update the path to the dataset in `RecommendationSystem/Read_CSV/read_csv.py`



## Maintainers:
* Anonymous

## Contributors:
* Anonymous

## License:
Copyright(c) 2025 - today Anonymous

Distributed under the [MIT License](LICENSE)

