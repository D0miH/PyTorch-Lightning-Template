FROM pytorch/pytorch:1.10.0-cuda11.3-cudnn8-devel

ARG WANDB_KEY

ENV LC_ALL=C.UTF-8
ENV LANG=C.UTF-8

# set the working directory and copy everything to the docker file
WORKDIR ./
COPY ./requirements.txt ./

RUN apt-get update
RUN apt-get -y upgrade
RUN conda install --rev 1
RUN conda install python=3.8.10
RUN conda install pytorch==1.10.0 torchvision==0.11.1 torchaudio cudatoolkit=11.3 -c pytorch
RUN pip install -r requirements.txt -f https://download.pytorch.org/whl/torch_stable.html

RUN if [ -z "$WANDB_KEY" ] ; then echo WandB API key not provided ; else wandb login "$WANDB_KEY"; fi
