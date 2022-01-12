#!/bin/bash
IMAGE_NAME=pytorch_lightning_template
CONTAINER_NAME=pytorch_lightning_template
DEVICES=0
MOUNTING_FILE=""

POSITIONAL=()
while [[ $# -gt 0 ]]
do
key="$1"

case $key in
    -i|--image)
    IMAGE_NAME="$2"
    shift # passed argument
    shift # passed value
    ;;
    -n|--name)
    CONTAINER_NAME="$2"
    shift
    shift
    ;;
    -d|--devices)
    DEVICES="$2"
    shift
    shift
    ;;
    -m|--mount_file)
    MOUNTING_FILE="$2"
    shift
    shift
    ;;
esac
done
set -- "${POSITIONAL[@]}"

DEVICE=$(echo $DEVICE | tr -d '"')

ADDITIONAL_MOUNTING_COMMAND=""
if [ -n "${MOUNTING_FILE}" ] ; then
  while read -r line; do
    [[ "$line" =~ ^#.*$ ]] && continue
    ADDITIONAL_MOUNTING_COMMAND+="-v \$(pwd)$line:workspace$line"
  done < "$MOUNTING_FILE"
fi

echo "----------Running the following command:----------"
echo "docker run --rm --name "${CONTAINER_NAME}" --gpus '\""device=$DEVICES"\"' -v \$(pwd):/workspace "${ADDITIONAL_MOUNTING_COMMAND}" -itd "${IMAGE_NAME}" bash"
echo "--------------------------------------------------"
eval "docker run --rm --name "${CONTAINER_NAME}" --gpus '\""device=$DEVICES"\"' -v $(pwd):/workspace "${ADDITIONAL_MOUNTING_COMMAND}" -itd "${IMAGE_NAME}" bash"