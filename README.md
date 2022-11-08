# HCMC_AIChallenge Team BacHNG Solution

### How to run

git clone the project

run the following commands

```console
cd aichallenge_frontend
docker-compose up
```

open a new terminal and run the following commands
```console
cd SenmaticSearchCLIP
docker build --tag python-docker .
docker run -p 5000:5000 python-docker

```

All the keyframes from the HCMC_AIChallenge2022 organizer have to be downloaded and put into a folder called "KeyFramesC00_V00" in the "aichallenge_frontend/public" for this to work properly.
