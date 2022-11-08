FROM pytorch/pytorch:1.11.0-cuda11.3-cudnn8-runtime

# these two lines are mandatory
WORKDIR /gridai/project
COPY . .

# any RUN commands you'd like to run
# use this to install dependencies
RUN apt-get update && apt-get install -y ffmpeg \
        libsm6 \
        libxext6

RUN pip3 install -r requirements.txt
