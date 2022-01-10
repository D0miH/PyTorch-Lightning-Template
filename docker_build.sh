#!/bin/bash

# default name for the container
NAME=pytorch_lightning_template
WANDB_KEY=""

POSITIONAL=()
while [[ $# -gt 0 ]]
do
key="$1"

case $key in
    -n|--name)
    NAME="$2"
    shift # passed argument
    shift # passed value
    ;;
    -w|--wandb)
    WANDB_KEY="$2"
    shift
    shift
    ;;
esac
done
set -- "${POSITIONAL[@]}"

echo "Image name: ${NAME}"
echo "WandB API key: ${WANDB_KEY}"

docker build -t "${NAME}" --build-arg WANDB_KEY=${WANDB_KEY} .