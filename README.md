# PyTorch-Lightning-Template
A PyTorch-Lightning template that supports yapf autoformatting and includes a docker container to run the code.

# Setup

## Docker
To build the Docker container run: 
```bash
./docker_build.sh -n <image-name> -w <wandb-key>
```

## Local Setup
To set up the project locally to use `yapf` and `pre-commit` you have to first install all dependencies using `conda` and `pip`:
```bash
conda install pytorch torchvision torchaudio cudatoolkit=11.3 -c pytorch
pip install -r requirements.txt
```

Then you have to set up 'pre-commit` to run yapf each time before committing:
```bash
pre-commit install
```

That's it!🎊️ 
Now you are setup and ready to go!

# Start Docker Container
To start the Docker container run the following command:
```bash
./docker_run.sh -n <container_name> -i <image_name> -d <comma_separated_device_ids>
# e.g. ./docker_run.sh -n pytorch_lightning_template -i pytorch_lightning_template -d "0,1"
```
or
```bash
docker run --rm --name <container-name> --gpus '"device=<device-ids>"' -v $(pwd):/workspace  -it <image-name> bash
```
