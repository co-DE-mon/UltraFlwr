# need cuda 12.4 and python 3.10
FROM nvcr.io/nvidia/pytorch:24.05-py3

ARG USER_ID
ARG GROUP_ID
ARG USER

RUN echo "Building with user: "$USER", user ID: "$USER_ID", group ID: "$GROUP_ID

RUN addgroup --gid $GROUP_ID $USER && \
    adduser --disabled-password --gecos '' --uid $USER_ID --gid $GROUP_ID $USER

# set working directory
WORKDIR /nfs/home/$USER

# Install dependencies
RUN apt update && \
    apt install -y lsof && \
    apt clean && rm -rf /var/lib/apt/lists/*

# Install Python packages
RUN pip install prettytable==3.12.0 flwr==1.14.0 ultralytics==8.3.53 opencv-python==4.8.0.74
