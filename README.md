# AccelerQ

This repository is a public version of the [KCL_QAGC](https://github.com/Connorpl/KCL_QAGC) project, which is actively under development.

## 1. Project Overview

Evaluation was done with two QE solver implementations. While the QCELS implementation is supplied with this Git Repository, you will have to pull the ADPT-QSCI code (the part that requires no changes) from the original Git Repository.

### Team

**King's College London (Informatics Department):**

- Avner Bensoussan
- Karine Even-Mendoza
- Sophie Fortz

**King's College London (Physics Department):**

- Elena Chachkarova
- Connor Lenihan

## 2. Experiments

Ensure your system meets the requirements, then follow the instructions to reproduce the figures in Section 7 (Results).

Hardware requirements: You will need a Unix machine with 10 GB HD free space and 8 GB of RAM, at least. You might need to set a swap. Operating System: Tested on Ubuntu.

### 2.1 Software and Packages:

You will need to install the following:

If using a Docker image or building with Docker
- Docker

or else
- Python 3.10.12
- Several Python and Unix packages, detailed below

We recommend using Docker, at least at first.

### 2.2 Setting UP

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

### 2.3 Python Packages

Please take a look at the dependencies in the requirements.txt file as well as some ad-hoc solutions in the Dockerfile and the Docker image. 

Python package installations tend to break easily (in general). Please contact us if you require further help.

### 2.4 Setup Troubleshooting

You might need to create a swap file to run the ADPT-QSCI and QCELS with 20+ qubits.
```
./AccelerQ/scripts/0-swap-setup.sh <YOUR-HOME-DIR>
```

If you have a permission issue running Docker, you can run this script to try to solve it:
```
./AccelerQ/scripts/docker_troubleshooting.sh
```

### 2.5 Hardware Specifications

- Architectures: x86, ARM

## 3. Reproduce OOPSLA 2025 Evaluation:

### 3.1 Table 1 - Analysis of Implementations' Hyperparameters

Table 1 was extracted manually. For reproducibility and generalisability, we explain below the process. You can go directly to 3.2 if you wish to continue with the execution of the artifact.

These parameters are given as input with the tested/optimised quantum eigensolver (QE) implementations. We describe them here to allow a replication study on similar/same QE implementations. We give the GitHub links of the QE implementations analysed in this study:

- (Use-Case-1) QCELS, which operates directly on Hamiltonian [1,2]
- (Use-Case-2) ADPT-QSCI, which transforms the Hamiltonian to a quantum circuit [3,4,5].

**YOU DO NOT NEED TO DOWNLOAD THE DATA of QCELS (Use-Case-1) and ADPT-QSCI (Use-Case-2) UNLESS YOU WISH TO REPRODUCE THE RESULTS OF THE PAPER FOR REPLICATION STUDY**. 

**WHY?** Because the scripts here already copy the data from both repositories into the right place before execution and already include the hyperparameters as arguments of a Python function. You can, however, check that all the arguments are related to a hyperparameter for each of the quantum implementations. For other QE implementations, if not exposing the hyperparameters as arguments, it is required to edit the code.

For example, this [QE implementation](https://github.com/morim3/DirectOptimizingQSCI/blob/main/problem/answer.py#L494), instead of having it encoded directly:
```
            atol=1e-6,
            final_sampling_shots_coeff=1,
            max_num_converged=2000,
```
You need to have it as a vector of arguments as input. Another example of a [QE implementation](https://github.com/Louisanity/SUN-Qsim/blob/main/problem/answer.py#L237), where the hyperparmeters are typed directly:
```
            depth=2,
            optimizer=SLSQP,
            max_steps=100,
            init_param=None
```
in both cases, most of the hyperparameters are relatively simple to spot and move outside the function to be arguments.

#### QCELS (Use-Case-1)

An implementation of the algorithm [2] is provided in [QCELS/QCELS_answer.py](https://github.com/karineek/AccelerQ/blob/main/QCELS/QCELS_answer.py), copied as is from [1].

**Note on this implementation:** it follows the Hamiltonian model. That is, rather than using discrete gates, the Hamiltonian model computes by evolving the quantum system continuously in time under a time-dependent Hermitian matrix $$H(t)$$. This evolution is governed by **Schrödinger’s equation**:

$$
i \hbar \frac{d}{dt} |\psi(t)\rangle = H(t) |\psi(t)\rangle
$$

[1] Connorpl. Accessed: July 4, 2024. QCELS_for_QAGC. [https://github.com/Connorpl/QCELS_for_QAGC].

[2] Zhiyan Ding and Lin Lin. 2023. Even Shorter Quantum Circuit for Phase Estimation on Early Fault-Tolerant Quantum Computers with Applications to
Ground-State Energy Estimation. PRX Quantum 4 (May 2023), 020331. Issue 2. [https://doi.org/10.1103/PRXQuantum.4.020331]

#### ADPT-QSCI (Use-Case-2)

An [implementation](https://github.com/QunaSys/quantum-algorithm-grand-challenge-2024/blob/main/problem/example_adaptqsci.py) of the algorithm [5] is provided in [3,4].

**Note on this implementation:** it follows the circuit model.

As part of the Docker image, we clone the original repository as we use the code and the data there for training:
```
git clone https://github.com/QunaSys/quantum-algorithm-grand-challenge-2024.git
```
and then copy the relevant data to utils and hamiltonian folders.

```
cp -r quantum-algorithm-grand-challenge-2024/utils .
cp quantum-algorithm-grand-challenge-2024/hamiltonian/* hamiltonian/
cp quantum-algorithm-grand-challenge-2024/problem/first_answer.py src/
```
[3] QunaSys. February 1, 2023. Quantum Algorithm Grand Challenge 2023 (QAGC2023). [https://github.com/QunaSys/quantum-algorithm-grandchallenge-2023](https://github.com/QunaSys/quantum-algorithm-grand-challenge-2023).

[4] QunaSys. February 1, 2024. Quantum Algorithm Grand Challenge 2024 (QAGC2024). [[https://github.com/QunaSys/quantum-algorithm-grandchallenge-2024](https://github.com/QunaSys/quantum-algorithm-grand-challenge-2024)].

[5] Keita Kanno, Masaya Kohda, Ryosuke Imai, Sho Koh, Kosuke Mitarai, Wataru Mizukami, and Yuya O. Nakagawa. 2023. Quantum-Selected
Configuration Interaction: classical diagonalization of Hamiltonians in subspaces selected by quantum computers. arXiv:2302.11320 [quant-ph]
[https://arxiv.org/abs/2302.11320]

### 3.2 Table 2 - Tests for QE Implementation

This part is also written manually, unless the QE implementation already has a set of tests/unittests/assumptions to check against, in which case, you can use these, of course.

The only assumption we used that was given here is that the number of shots in total is no more than 10^7, the rest we wrote.

- (Use-Case-1) QCELS tests are [here](https://github.com/karineek/AccelerQ/blob/main/src/kcl_tests_qcels.py);
- (Use-Case-2) ADPT-QSCI tests are [here](https://github.com/karineek/AccelerQ/blob/main/src/kcl_tests_adapt_vqe.py).

### 3.3 Data of 6.2 Experimental Setup

The data is taken from the resources as described in the paper. For the experiments, you can find the data in the [Hamiltonian folder](https://github.com/karineek/AccelerQ/tree/main/hamiltonian). 
The Dockerfile script copies the Hamiltonian of [[4](https://github.com/QunaSys/quantum-algorithm-grand-challenge-2024/tree/main/hamiltonian)], too.

Using these Hamiltonians, you can perform the first step: Data Augmentation. By running the first stage for each QE implementation:

```
python3 kcl_QCELS_stage_1.py
python3 kcl_adapt_vqe_stage_1.py
```
The scripts are pretty similar, but contain imports of the respective QE implementation, where you need to change the small system mined:
```
    # inputs
    folder_path = "../hamiltonian/"
    prefix = "16qubits_05" #  >>>>>>>>>>>>>>>> Change only this!
    file_name = prefix + ".data"
    X_file = prefix + ".X.data"
    Y_file = prefix + ".Y.data"
```

#### Partial Evaluation

Some systems can take a long time to mine. We create a short script that does it for 2-3 systems.

#### Full-Evaluation

This script mines data from all small systems, but it might take several days to run.



### 3.3 Figure 5 - Hyperparameters' value distribution

We checked 
