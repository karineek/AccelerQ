# Use the official Python image with Python 3.10 as a base image
FROM python:3.10

# Switch to root user for system-level installations
USER root

# Create a new user to work in
RUN rm /bin/sh && ln -s /bin/bash /bin/sh
RUN useradd -ms /bin/bash kclq

# Update system and install necessary packages
RUN apt-get update
RUN apt-get install -y software-properties-common
RUN apt-get update && apt-get install -y apt-utils
RUN apt-get -y update \
    && apt-get install -y build-essential git m4 scons zlib1g zlib1g-dev \
        libprotobuf-dev protobuf-compiler libprotoc-dev libgoogle-perftools-dev \
        python3-dev libboost-all-dev pkg-config libssl-dev \
        libpng-dev libpng++-dev libhdf5-dev \
        python3-pip python3-venv automake
RUN apt-get -y install libcurl4-openssl-dev libcurl4-doc libidn-dev libkrb5-dev libldap2-dev librtmp-dev libssh2-1-dev
RUN apt-get install -y cmake libopenblas-dev

  USER kclq
  WORKDIR /home/kclq
  RUN curl -O https://www.python.org/ftp/python/3.10.12/Python-3.10.12.tgz \
    # Create a dummy file to signal Docker to cache the download
    && touch Python-3.10.12.tgz.downloaded
  RUN tar -xvzf Python-3.10.12.tgz
  WORKDIR /home/kclq/Python-3.10.12
  RUN ./configure --enable-optimizations
  RUN make

USER root
RUN make install
ENV PATH="/home/kclq/.local/bin:${PATH}"


# Install system dependencies and Rust
RUN apt-get update && \
    apt-get install -y curl && \
    curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y && \
    . "$HOME/.cargo/env" && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

# Set up Cargo path for future steps
ENV PATH="/root/.cargo/bin:${PATH}"


  # Clone the repository from GitHub
  USER kclq
  WORKDIR /home/kclq
  RUN git clone https://github.com/QunaSys/quantum-algorithm-grand-challenge-2024.git
  # Install required packages
  RUN pip3 install --upgrade pip --no-warn-script-location
  RUN pip3 install requests~=2.28.0 --no-warn-script-location
  RUN pip3 install pip pipenv --upgrade
  RUN pip3 install pipenv

  # Get our repository
  RUN git clone git@github.com:karineek/AccelerQ.git

  # Install Jupyter Notebook
  RUN pip3 install jupyter
  RUN pip3 install pip pipenv --upgrade
  RUN pip3 install -r requirements.txt
  RUN pip3 install xgboost
  RUN pip3 install pytket quri-parts-tket
  RUN pip3 install openfermion
  RUN pip3 install quri-parts-openfermion
  RUN pip3 install quri-parts-qulacs
  RUN pip3 install qiskit
  RUN pip3 install juliacall
  RUN pip3 install quri-parts-itensor
  RUN pip3 install quri-parts
  RUN pip3 install stopit
  
  # Install Julia things
  RUN python3 STAB.py
  RUN pip3 install openfermion # last update!

  # Starts the docker
  USER root
  WORKDIR /home/kclq/AccelerQ/
