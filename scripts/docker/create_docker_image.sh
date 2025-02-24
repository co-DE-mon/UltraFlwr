#!/usr/bin/env bash

# navigate to directory
SCRIPTPATH="$( cd "$(dirname "$0")" ; pwd -P )"
cd $SCRIPTPATH

cd ../../

docker_tag=<YOUR_DOCKERHUB_USERNAME>/<YOUR_IMAGE_TAG>


docker build . -f Dockerfile \
 --network=host \
 --tag ${docker_tag} \
 --build-arg USER_ID=$(id -u) \
 --build-arg GROUP_ID=$(id -g) \
 --build-arg USER=${USER}

docker push ${docker_tag}
