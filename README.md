# AccelerQ

This repository is a public version of the [KCL_QAGC](https://github.com/Connorpl/KCL_QAGC) project, which is actively under development.

## Project Overview

Evaluation was done with two QE solver implementations. While the QCELS implementation is supplied with this Git Repository, you will have to pull the ADPT-QSCI code (the part that required no changes) from the original Git Repository.

### Team

**King's College London (Informatics Department):**

- Avner Bensoussan
- Karine Even-Mendoza
- Sophie Fortz

**King's College London (Physics Department):**

- Elena Chachkarova
- Connor Lenihan

## Experiments

Ensure your system meets the requirements, then follow the instructions to reproduce the figures in Section 7 (Results).

Hardware requirements: You will need a Unix machine with 10 GB HD free space and 8 GB of RAM, at least. You might need to set a swap. Operating System: Tested on Ubuntu.

### Software and Packages:

You will need to install the following:

If using Docker image or building with Docker
- Docker

or else
- Python 3.10.12
- Several Python and Unix packages, detailed below

We recommend using Docker, at least at first.

### Setting UP

Get the code and build docker:
```
git clone git@github.com:karineek/AccelerQ.git
cd AccelerQ
docker build -t dockerfile .
```
then run:
```
docker run -dt dockerfile
```

**SKIP THIS PART**, unless you wish to install the tool locally, outside DOCKER (not recommended, unless developing new parts).

Before starting, install Python3.10.12 as the default on your system.
```
sudo apt-get update
sudo apt-get install -y software-properties-common
sudo apt-get update && apt-get install -y apt-utils
sudo apt-get -y update \
    && apt-get install -y build-essential git m4 scons zlib1g zlib1g-dev \
        libprotobuf-dev protobuf-compiler libprotoc-dev libgoogle-perftools-dev \
        python3-dev libboost-all-dev pkg-config libssl-dev \
        libpng-dev libpng++-dev libhdf5-dev \
        python3-pip python3-venv automake
sudo apt-get -y install libcurl4-openssl-dev libcurl4-doc libidn-dev libkrb5-dev libldap2-dev librtmp-dev libssh2-1-dev
sudo apt-get install -y cmake libopenblas-dev
```
get the tool:
```
git clone git@github.com:karineek/AccelerQ.git
cd AccelerQ/scripts/
```
then install the Python requirements
```
# Upgrade pip
pip3 install --upgrade pip --no-warn-script-location

# Core tools
pip3 install pipenv --upgrade
pip3 install jupyter
pip3 install stopit
pip3 install requests~=2.28.0 --no-warn-script-location

# Install from requirements file (if exists)
pip3 install -r requirements.txt

# ML libraries
pip3 install xgboost

# Quantum frameworks
pip3 install qiskit
pip3 install openfermion  # (installed only once)
pip3 install quri-parts
pip3 install quri-parts-openfermion
pip3 install quri-parts-qulacs
pip3 install quri-parts-tket
pip3 install quri-parts-itensor

# Optional Julia interface
pip3 install juliacall

# Run main script
python3 STAB.py
```

### Python Packages

Please take a look at the dependencies in the requirements.txt file.

### Setup Troubleshooting

You might need to create a swap file to run the ADPT-QSCI and QCELS with 20+ qubits.
```
./AccelerQ/scripts/0-swap-setup.sh <YOUR-HOME-DIR>
```

If you have a permission issue running Docker, you can run this script to try to solve it:
```
./AccelerQ/scripts/docker_troubleshooting.sh
```

### Hardware Specifications

- Architectures: x86, ARM

### Reproduce OOPSLA 2025 Evaluation:

TODO

## Additional Information about the Optimised Quantum Implementation and Data

We used two algorithms.

### QCELS

We wrote a version of the algorithm. See QCELS/QCELS_answer.py. This code is only for reference.

### ADPT-QSCI

Clone the original repository
```
git clone https://github.com/QunaSys/quantum-algorithm-grand-challenge-2024.git
```
and then copy the relevant data to utils and hamiltonian folders.

```
cp -r quantum-algorithm-grand-challenge-2024/utils .
cp quantum-algorithm-grand-challenge-2024/hamiltonian/* hamiltonian/
cp quantum-algorithm-grand-challenge-2024/problem/first_answer.py src/
```

