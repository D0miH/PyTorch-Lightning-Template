#!/bin/bash
IMAGE_NAME=clipping_privacy
CONTAINER_NAME=clipping_privacy
DEVICES=0
MOUNTING_FILE=""
SHM_SIZE="16G"
PORT_MAPPING=""

POSITIONAL=()
while [ $# -gt 0 ]
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
    --shm-size)
    SHM_SIZE="$2"
    shift
    shift
    ;;
    -m|--mount_file)
    MOUNTING_FILE="$2"
    shift
    shift
    ;;
    -p)
    PORT_MAPPING="$2"
    shift
    shift
    ;;
esac
done
set -- "${POSITIONAL[@]}"

DEVICE_COMMAND=""
if [[ $DEVICES =~ ^[0-9]+$ ]] ; then
  DEVICE=$(echo "$DEVICES" | tr -d '"')
  DEVICE_COMMAND=\""device=$DEVICE"\"
else
  DEVICE_COMMAND='all'
fi


ADDITIONAL_MOUNTING_COMMAND=""
if [ -n "${MOUNTING_FILE}" ] ; then
  while read -r line; do
    [[ "$line" =~ ^#.*$ ]] && continue
    ADDITIONAL_MOUNTING_COMMAND+=" -v \$(pwd)$line:/workspace$line"
  done < "$MOUNTING_FILE"
fi

PORT_MAPPING_CMD=""
if [ -n "${PORT_MAPPING}" ] ; then
  PORT_MAPPING_CMD="-p ${PORT_MAPPING} "
fi

echo "----------Running the following command:----------"
echo "docker run --rm --shm-size ${SHM_SIZE} --name ${CONTAINER_NAME} --gpus '${DEVICE_COMMAND}' -v \$(pwd):/workspace${ADDITIONAL_MOUNTING_COMMAND} ${PORT_MAPPING_CMD}-itd ${IMAGE_NAME} bash"
echo "--------------------------------------------------"
eval "docker run --rm --shm-size ${SHM_SIZE} --name ${CONTAINER_NAME} --gpus '${DEVICE_COMMAND}' -v \$(pwd):/workspace${ADDITIONAL_MOUNTING_COMMAND} ${PORT_MAPPING_CMD}-itd ${IMAGE_NAME} bash"
