IMAGE_NAME=pytorch_lightning_template
CONTAINER_NAME=pytorch_lightning_template
DEVICES=0

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
esac
done
set -- "${POSITIONAL[@]}"

DEVICE=$(echo $DEVICE | tr -d '"')

eval "docker run --rm --name "${CONTAINER_NAME}" --gpus '\""device=$DEVICES"\"' -v $(pwd):/workspace -itd "${IMAGE_NAME}" bash"