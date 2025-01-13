# What-Do-Other-People-Think-About-This---Recommendation-Model

This repository contains the code for the implementation of our comment recommendation model for the paper `What Do Other People Think About This? Recommending Relevant and Diverse User Comments in Comment Sections`.

The model does not recommend comments based on a user's past interests or behavior, but rather based on the comment he or she is interested in. It uses a "recommendation score" to identify relevant comments and sort them accordingly. It also uses various machine learning models to identify comments with different viewpoints on the topic the user is currently interested in.




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



## Maintainers:
* Anonymous

## Contributors:
* Anonymous

## License:
Copyright(c) 2025 - today Anonymous

Distributed under the [MIT License](LICENSE)

