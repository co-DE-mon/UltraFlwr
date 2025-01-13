#!/usr/bin/env bash

docker_tag=yang20180924/${USER}ultraflwr:latest


docker build . -f Dockerfile \
 --network=host \
 --tag ${docker_tag} \
 --build-arg USER_ID=$(id -u) \
 --build-arg GROUP_ID=$(id -g) \
 --build-arg USER=${USER}

docker push ${docker_tag}
