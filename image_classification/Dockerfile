FROM tensorflow/tensorflow:2.5.0-gpu

WORKDIR /research/resnet
# Mount data into the docker
ADD ./tensorflow2 ${WORKDIR}

# Add vim to container for debugging
RUN rm /etc/apt/sources.list.d/cuda.list
RUN rm /etc/apt/sources.list.d/nvidia-ml.list
RUN apt-key del 7fa2af80
RUN apt-get update && apt-get install -y --no-install-recommends wget
RUN wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu1804/x86_64/cuda-keyring_1.0-1_all.deb
RUN dpkg -i cuda-keyring_1.0-1_all.deb
RUN apt-get install -y vim

ENTRYPOINT ["/bin/bash"]