# PyTorch-Lightning-Template
A PyTorch-Lightning template that supports yapf autoformatting and includes a docker container to run the code.

# Setup

# Setup Using Docker
## Build the Docker Image
To build the Docker image run: 
```bash
./docker_build.sh -n <image-name> -w <wandb-key>
```

## Start Docker Container
To start the Docker container run the following command:
```bash
./docker_run.sh -n <container_name> -i <image_name> -d <comma_separated_device_ids>
# e.g. ./docker_run.sh -n pytorch_lightning_template -i pytorch_lightning_template -d "0,1"
```
or
```bash
docker run --rm --name <container-name> --gpus '"device=<device-ids>"' -v $(pwd):/workspace  -it <image-name> bash
```

That's it!🎊️  
Now you are setup and ready to develop using Docker!

# Local Setup
If you don't want to use Docker and instead want to set up the project locally to use `yapf` and `pre-commit` you have to first install PyTorch:
```bash
conda install pytorch torchvision torchaudio pytorch-cuda=11.7 -c pytorch -c nvidia
```
Then you have to install all other dependencies:
```bash
pip install -r requirements.txt
```

Finally, you have to set up 'pre-commit` to run yapf each time before committing:
```bash
pre-commit install
```

That's it!🎊️ 
Now you are setup and ready to go!


