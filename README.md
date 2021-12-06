# PyTorch-Lightning-Template

# Setup

## Docker
To build the Docker container run: 
```bash
./docker_buid.sh -n <image-name> -w <wandb-key>
```

## Local Setup
To set up the project locally to use `yapf` and `pre-commit` you have to first install all dependencies:
```bash
pip install -r requirements.txt -f https://download.pytorch.org/whl/torch_stable.html
```

Then you have to setup 'pre-commit` to run yapf each time before commiting:
```bash
pre-commit install
```

That's it!🎊️ 
Now you are setup and ready to go!