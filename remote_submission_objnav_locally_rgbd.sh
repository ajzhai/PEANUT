#!/usr/bin/env bash

DOCKER_NAME="remote_submission"

while [[ $# -gt 0 ]]
do
key="${1}"

case $key in
      --docker-name)
      shift
      DOCKER_NAME="${1}"
	  shift
      ;;
    *) # unknown arg
      echo unknown arg ${1}
      exit
      ;;
esac
done

docker run -v $(pwd)/habitat-challenge-data:/habitat-challenge-data \
    -v $(realpath habitat-challenge-data/data/scene_datasets/hm3d):/habitat-challenge-data/data/scene_datasets/hm3d \
    -v $(pwd)/data:/data\
    -v $(pwd)/Stubborn:/Stubborn\
    -v $(realpath habitat-challenge-data/data/scene_datasets/hm3d):/data/scene_datasets/hm3d \
    --gpus='all' \
    -e "AGENT_EVALUATION_TYPE=local" \
    -e "TRACK_CONFIG_FILE=/challenge_objectnav2022.local.rgbd.yaml" \
    ${DOCKER_NAME}\

