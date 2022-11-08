# HCMC_AIChallenge Team BacHNG Solution

### How to run

git clone the project

run the following commands

"""console
cd aichallenge_frontend
docker-compose up

cd ../
cd SenmaticSearchCLIP
docker build --tag python-docker .
docker run -p 5000:5000 python-docker

"""
