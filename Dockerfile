# Use the official Python image with Python 3.10 as a base image
FROM python:3.10

# Switch to root user for system-level installations
USER root

# Create a new user to work in
RUN rm /bin/sh && ln -s /bin/bash /bin/sh
RUN useradd -ms /bin/bash kclq

ENV PIP_USER=no

# Update system and install necessary packages
RUN apt-get update
RUN apt-get install -y software-properties-common
RUN apt-get update && apt-get install -y apt-utils
RUN apt-get -y update \
    && apt-get install -y build-essential git m4 scons zlib1g zlib1g-dev \
        libprotobuf-dev protobuf-compiler libprotoc-dev libgoogle-perftools-dev \
        python3-dev libboost-all-dev pkg-config libssl-dev \
        libpng-dev libpng++-dev libhdf5-dev \
        python3-pip python3-venv automake vim nano
RUN apt-get -y install libcurl4-openssl-dev libcurl4-doc libidn-dev libkrb5-dev libldap2-dev librtmp-dev libssh2-1-dev
RUN apt-get install -y cmake libopenblas-dev

  #  USER kclq
  WORKDIR /home/kclq
  RUN curl -O https://www.python.org/ftp/python/3.10.12/Python-3.10.12.tgz \
    # Create a dummy file to signal Docker to cache the download
    && touch Python-3.10.12.tgz.downloaded
  RUN tar -xvzf Python-3.10.12.tgz
  WORKDIR /home/kclq/Python-3.10.12
  RUN ./configure --enable-optimizations
  RUN make

#USER root
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

# Try to fix Python again...
RUN ln -sf /usr/local/bin/python3.10 /usr/bin/python3 && \
    ln -sf /usr/local/bin/pip3.10 /usr/bin/pip3


  # Clone the repository from GitHub
  #  USER kclq
  WORKDIR /home/kclq
  RUN git clone https://github.com/QunaSys/quantum-algorithm-grand-challenge-2024.git
  # Install required packages
  RUN /usr/local/bin/python3.10 -m pip install --upgrade pip --no-warn-script-location
  RUN /usr/local/bin/python3.10 -m pip install requests~=2.28.0 --no-warn-script-location
  RUN /usr/local/bin/python3.10 -m pip install pip pipenv --upgrade
  RUN /usr/local/bin/python3.10 -m pip install pipenv

  # Get our repository
  RUN git clone https://github.com/karineek/AccelerQ.git

  # Add data and code needed
  RUN cp -r /home/kclq/quantum-algorithm-grand-challenge-2024/utils /home/kclq/AccelerQ/
  RUN cp /home/kclq/quantum-algorithm-grand-challenge-2024/hamiltonian/* /home/kclq/AccelerQ/hamiltonian/

  # Install Jupyter Notebook
  WORKDIR /home/kclq/AccelerQ/
  RUN /usr/local/bin/python3.10 -m pip install jupyter
  RUN /usr/local/bin/python3.10 -m pip install pip pipenv --upgrade
  RUN /usr/local/bin/python3.10 -m pip install -r requirements.txt
  RUN /usr/local/bin/python3.10 -m pip install xgboost
  RUN /usr/local/bin/python3.10 -m pip install pytket quri-parts-tket
  RUN /usr/local/bin/python3.10 -m pip install openfermion
  RUN /usr/local/bin/python3.10 -m pip install quri-parts-openfermion
  RUN /usr/local/bin/python3.10 -m pip install quri-parts-qulacs
  RUN /usr/local/bin/python3.10 -m pip install qiskit


  # Going into Julia none-sense
  ENV JULIA_VERSION=1.11.1
  RUN curl -fsSL https://julialang-s3.julialang.org/bin/linux/x64/1.11/julia-${JULIA_VERSION}-linux-x86_64.tar.gz \
    | tar -xz -C /opt && \
    ln -s /opt/julia-${JULIA_VERSION}/bin/julia /usr/local/bin/julia
  ENV JULIA_PYTHONCALL_EXE=/usr/local/bin/julia


  RUN /usr/local/bin/python3.10 -m pip install juliacall
  RUN /usr/local/bin/python3.10 -m pip install quri-parts-itensor
  RUN /usr/local/bin/python3.10 -m pip install quri-parts
  RUN /usr/local/bin/python3.10 -m pip install stopit
  
  # Install Julia things
  WORKDIR /home/kclq/AccelerQ/scripts/
  RUN cp /home/kclq/AccelerQ/src/first_answer.py /home/kclq/AccelerQ/scripts/
  RUN python3 STAB.py || true
  RUN /usr/local/bin/python3.10 -m pip install openfermion # last update!

  # A fix due to old version no longer working 
  RUN /usr/local/bin/python3.10 -m pip install juliacall==0.9.23
  RUN /usr/local/bin/python3.10 -m pip install juliapkg==0.1.14
  RUN julia --project=/root/.julia/environments/pyjuliapkg \
        -e 'using Pkg; Pkg.add(PackageSpec(name="ITensors", version="0.7.1"))'
  RUN /usr/local/bin/python3.10 -m pip install quri-parts-itensor==0.20.3

  # Starts the docker
  # USER root
  WORKDIR /home/kclq/AccelerQ/src/
