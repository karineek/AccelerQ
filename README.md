# AccelerQ

This repository is a public version of the [KCL_QAGC](https://github.com/Connorpl/KCL_QAGC) project, which is actively under development.

## Project Overview

Evaluation was done with two QE solver implementations. While the QCELS implementation is supplied with this Git Repository, you will have to pull the ADPT-QSCI code (the part that requires no changes) from the original Git Repository.

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

If using a Docker image or building with Docker
- Docker

or else
- Python 3.10.12
- Several Python and Unix packages, detailed below

We recommend using Docker, at least at first.

### Setting UP

Get the code and build Docker:
```
git clone git@github.com:karineek/AccelerQ.git
cd AccelerQ
docker build --no-cache -t dockerfile .
```
then run:
```
docker run -it dockerfile /bin/bash
```

**SKIP THIS PART**, unless you wish to install the tool locally, outside DOCKER (not recommended, unless developing new parts).

Before starting, install Python 3.10.12 as the default on your system.
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

Please take a look at the dependencies in the requirements.txt file as well as some ad-hoc solutions in the Dockerfile and the Docker image. 

Python package installations tend to break easily (in general). Please contact us if you require further help.

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

Table 1. These are parameters that are given as input with the tested/optimised QE implementation.

In this paper, we used two quantum eigensolver (QE) implementations: 
- (Use-Case-1) QCELS, which operates directly on Hamiltonian
- (Use-Case-2) ADPT-QSCI, which transforms the Hamiltonian to a quantum circuit.

### QCELS (Use-Case-1)

A reference implementation of the algorithm is provided in QCELS/QCELS_answer.py, copied as is from [1], based on the original method described in [2].

Note on this implementation: it follows the Hamiltonian model. That is, rather than using discrete gates, the Hamiltonian model computes by evolving the quantum system continuously in time under a time-dependent Hermitian matrix $$H(t)$$. This evolution is governed by **Schrödinger’s equation**:

$$
i \hbar \frac{d}{dt} |\psi(t)\rangle = H(t) |\psi(t)\rangle
$$

[1] Connorpl. Accessed: July 4, 2024. QCELS_for_QAGC. [https://github.com/Connorpl/QCELS_for_QAGC].

[2] Zhiyan Ding and Lin Lin. 2023. Even Shorter Quantum Circuit for Phase Estimation on Early Fault-Tolerant Quantum Computers with Applications to
Ground-State Energy Estimation. PRX Quantum 4 (May 2023), 020331. Issue 2. [https://doi.org/10.1103/PRXQuantum.4.020331]

### ADPT-QSCI (Use-Case-2)

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

