FROM nvidia/cuda:11.8.0-cudnn8-devel-ubuntu18.04

RUN apt update && apt upgrade -y
RUN apt install -y python3.8
RUN apt install -y python3-pip
RUN rm /usr/bin/python3
RUN ln -s python3.8 /usr/bin/python3
RUN DEBIAN_FRONTEND=noninteractive apt-get install -y python3-opencv

RUN apt install -y git vim unzip curl openssh-client
RUN apt install python-setuptools python3.8-dev python3-wheel build-essential -y
RUN apt update && apt upgrade -y

# PIP packages
RUN python3 -m pip install -U pip
RUN python3 -m pip install -U setuptools
RUN python3 -m pip install -U torch torchvision --index-url https://download.pytorch.org/whl/cu118

# Copy files
COPY ./ /lightning-pose
WORKDIR /lightning-pose

# Installing dependencies
RUN python3 -m pip install -e .