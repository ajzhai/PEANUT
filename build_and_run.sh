#!/usr/bin/env bash

DOCKER_NAME="peanut"

DOCKER_BUILDKIT=1 docker build . --build-arg INCUBATOR_VER=$(date +%Y%m%d-%H%M%S) --file peanut.Dockerfile -t ${DOCKER_NAME}

docker run -v $(pwd)/habitat-challenge-data:/habitat-challenge-data \
    -v $(realpath habitat-challenge-data/data/scene_datasets/hm3d):/habitat-challenge-data/data/scene_datasets/hm3d \
    -v $(realpath habitat-challenge-data/data/scene_datasets/hm3d):/data/scene_datasets/hm3d \
    -v $(pwd)/data:/data\
    -v $(pwd)/nav:/nav\
    --gpus='all' \
    -e "AGENT_EVALUATION_TYPE=local" \
    -e "TRACK_CONFIG_FILE=/challenge_objectnav2022.local.rgbd.yaml" \
    --ipc=host \
    ${DOCKER_NAME}\
