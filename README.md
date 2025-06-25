# AccelerQ

This repository is a public version of the [KCL_QAGC](https://github.com/Connorpl/KCL_QAGC) project, which is actively under development.

## 0. Artifact Evaluation

Read this carefully. **For the artifact evaluation:** Please go to [section 2.6](https://github.com/karineek/AccelerQ/blob/main/README.md#26-kick-the-tires) in this document to start. You can then later read this whole document when checking reproducibility. The commands for functionality will be marked (what you actually need to run). Note that this is artifact for quantum code optimisation, which likely takes months to run on a laptop. We, therefore, created a shorter version fit for a laptop. However, we gave the full details to run and develop further this platform if you have access to a GPU or a strong server. We mark these here with the label **fit for a laptop**. For example: [Phase 1 - Data Augmentation](https://github.com/karineek/AccelerQ/blob/main/README.md#partial-evaluation-fit-for-a-laptop).

The rest of this documentation goes beyond simply making the code **fit for a laptop**. Its purpose is to encourage writing code for quantum computing and to help other researchers get started more easily.

The structure of the artifact is:
```
AccelerQ-main/
├── Artifact_Experiments/   # Contains experiment automation scripts for the Artifact Evaluation
├── Dockerfile              # Docker setup for reproducibility
├── QCELS/                  # Original QCELS implementation
├── README.md               # Documentation
├── hamiltonian/            # Hamiltonian input files
├── models/                 # Pretrained model files
├── requirements.txt        # Dependency list for Python environment
├── scripts/                # Utility or orchestration scripts
└── src/                    # Source code for different VQE/QCELS pipelines

```

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

This section is fully **fit for a laptop**. However, the best is to start with Section 2.6, which quickly sets up the environment with a Docker image.

### 2.1 Software and Packages:

You will need to install the following:

If using a Docker image or building with Docker
- Docker

or else
- Python 3.10.12
- Python package setuptools<81 (we used setuptools 80.9.0)
- Several Python and Unix packages, detailed below

We recommend using Docker, at least at first.

### 2.2 Setting Up

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

### 2.6 Kick the Tires

```
python3 kcl_QCELS_stage_1.py
python3 kcl_adapt_vqe_stage_1.py
```

## 3. Reproduce OOPSLA 2025 Evaluation:

Most of the text here is for reproducibility. Please look for **fit for a laptop** tags to see what you can run on a laptop for artifact evaluation.

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






### 3.3 Phase 1 - Data Augmentation 

This creates the data of "Section 6.2 Experimental Setup".

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
These scripts create the data for the learning in the next phase. These scripts call the miner, which is in ```src/kcl_prepare_data.py```. 
The customisation is done via a wrapper function that is an argument for the miner function. The wrapper calls either of the algorithms with classical execution (classical flag is true).

These files in ```src``` folder are involved here:
```
AccelerQ-main
├── src
     ├── kcl_QCELS_stage_1.py, kcl_adapt_vqe_stage_1.py     # entry points for QCELS and adapt-VQE
     ├── kcl_util.py                                        # general-purpose helpers (e.g., process_file)
     ├── kcl_prepare_data.py                                # data preprocessing and feature extraction (miner)
     ├── kcl_util_qcels.py                                  # QCELS pipeline orchestration and wrappers
     ├── QCELS_answer_experiments.py                        # QCELS core logic (Wrapper, classical mode)
     ├── kcl_util_adapt_vqe.py                              # adapt-VQE pipeline orchestration and wrappers
     ├── first_answer.py                                    # adapt-VQE core logic (Wrapper, classical mode)
```
The data we mined is in the Zenodo Record: ADAPT-QSCI-data.tar.xz and QCELS-data.tar.xz.

To get yourself this data, you can run one of the versions below. We recommend running the short version **fit for a laptop**.

#### Partial Evaluation **fit for a laptop**

Some systems can take a long time to mine. We create a short script that automates the process for a few systems, each sampled 5 times.
```
cd Artifact_Experiments
chmod 777 phase_1_short.sh
./phase_1_short.sh
```
This script on X86 with 8 GB RAM ran in our Docker for 8 minutes. Data will be written into ```src``` folder:
```
root@08c84bd04541:/home/kclq/AccelerQ# ls -l src/*.npy
-rw-r--r-- 1 root root    128 Jun 25 10:23 src/02qubits_05.X.data.npy
-rw-r--r-- 1 root root    128 Jun 25 10:23 src/02qubits_05.Y.data.npy
-rw-r--r-- 1 root root   1968 Jun 25 10:22 src/02qubits_05A.X.data.npy
-rw-r--r-- 1 root root    208 Jun 25 10:22 src/02qubits_05A.Y.data.npy
-rw-r--r-- 1 root root  47408 Jun 25 10:51 src/04qubits_05.X.data.npy
-rw-r--r-- 1 root root    208 Jun 25 10:51 src/04qubits_05.Y.data.npy
-rw-r--r-- 1 root root  46928 Jun 25 10:50 src/04qubits_05A.X.data.npy
-rw-r--r-- 1 root root    208 Jun 25 10:50 src/04qubits_05A.Y.data.npy
-rw-r--r-- 1 root root  86128 Jun 25 10:54 src/06qubits_05.X.data.npy
-rw-r--r-- 1 root root   2128 Jun 25 10:54 src/06qubits_05.Y.data.npy
-rw-r--r-- 1 root root  74128 Jun 25 10:53 src/06qubits_05A.X.data.npy
-rw-r--r-- 1 root root   2128 Jun 25 10:53 src/06qubits_05A.Y.data.npy
-rw-r--r-- 1 root root 134128 Jun 25 10:57 src/08qubits_05.X.data.npy
-rw-r--r-- 1 root root   2128 Jun 25 10:57 src/08qubits_05.Y.data.npy
-rw-r--r-- 1 root root 122128 Jun 25 10:56 src/08qubits_05A.X.data.npy
-rw-r--r-- 1 root root   2128 Jun 25 10:56 src/08qubits_05A.Y.data.npy
```
The *A* are for QCELS and the other is for ADAPT-QSCI.

#### Full-Evaluation

This script mines data from **all** small systems, 660 samples per system, but it can take several days.
```
cd Artifact_Experiments
chmod 777 phase_1_full.sh
./phase_1_full.sh
```





### 3.4 Phase 2 - ML Model  

We use the data from the previous section (Phase 1), to train two models, one per quantum implementation.
```
python3 kcl_QCELS_stage_2.py
python3 kcl_adapt_vqe_stage_2.py
```
When using a GPU, you can edit this
```
cpu=1 # >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>> Change when using a GPU
```
to be falls, otherwise, this script should work on a CPU, but of course with lower performance.
For the artifact evaluation, you can train the model with the data and try things, but the results will be significantly different from the paper. 
If you have a GPU, you can retrain the model with the data from ADAPT-QSCI-data.tar.xz and QCELS-data.tar.xz.

This is phase 2, and these scripts are relevant to it:
```
AccelerQ-main
├── src
     ├── kcl_QCELS_stage_2.py, kcl_adapt_vqe_stage_2.py     # training entry points (QCELS/adapt-VQE)
     ├── kcl_util.py                                        # data loading, vectorisation, saving, and utility functions
     ├── kcl_train_xgb.py                                   # wraps XGBoost training (train, vec_to_fixed_size_vec, etc.)
```
Python libraries:
```
├── sklearn # model selection and regression metrics (train_test_split, mean_squared_error, etc.)
├── xgboost # XGBoost backend used via xgb.XGBRegressor
```
Note: No method-specific code (like first_answer.py or QCELS_answer_experiments.py) is called in Phase 2, except for data location and where to save the models.

#### Partial Evaluation **fit for a laptop**

Use the data you mined before, which is a very small sample, to train the model. This model quality will be low but is good to test on a CPU.
```
cd Artifact_Experiments
chmod 777 phase_2_short.sh
./phase_2_short.sh
```
This will create two models, one per QE implementation:
```
root@08c84bd04541:/home/kclq/AccelerQ# ls -l src/*.json
-rw-r--r-- 1 root root   237437 Jun 25 11:57 src/model_avqe_pre_xgb_28.json
-rw-r--r-- 1 root root 57326999 Jun 25 11:57 src/model_qcels_pre_xgb_28.json
```
and this will output something like after 4 minutes in our Docker on X86 machine with 8 GB RAM:
```
root@08c84bd04541:/home/kclq/AccelerQ/Artifact_Experiments# ./phase_2_short.sh 
>> Phase 2 starting...
total 264
-rw-r--r-- 1 root root   1968 Jun 25 11:56 02qubits_05A.X.data.npy
-rw-r--r-- 1 root root    208 Jun 25 11:56 02qubits_05A.Y.data.npy
-rw-r--r-- 1 root root  46928 Jun 25 11:56 04qubits_05A.X.data.npy
-rw-r--r-- 1 root root    208 Jun 25 11:56 04qubits_05A.Y.data.npy
-rw-r--r-- 1 root root  74128 Jun 25 11:56 06qubits_05A.X.data.npy
-rw-r--r-- 1 root root   2128 Jun 25 11:56 06qubits_05A.Y.data.npy
-rw-r--r-- 1 root root 122128 Jun 25 11:56 08qubits_05A.X.data.npy
-rw-r--r-- 1 root root   2128 Jun 25 11:56 08qubits_05A.Y.data.npy
>> Running kcl_QCELS_stage_2.py
>> Start Stage 2
>> Read ham ../hamiltonian/, 28qubits_01.data
28qubits_01
>> Start processing: 28qubits_01.data with qubits 28
>>>> adding ham of size 98700
>> Start Training
Load data from folder ../data/
Data loaded successfully from ../data/08qubits_05A.X.data.npy
>>> Stat. >>> Size of set loaded from: ../data/08qubits_05A.X.data.npy is: 225
Data loaded successfully from ../data/04qubits_05A.X.data.npy
>>> Stat. >>> Size of set loaded from: ../data/04qubits_05A.X.data.npy is: 9
Data loaded successfully from ../data/06qubits_05A.X.data.npy
>>> Stat. >>> Size of set loaded from: ../data/06qubits_05A.X.data.npy is: 225
Data loaded successfully from ../data/02qubits_05A.X.data.npy
>>> Stat. >>> Size of set loaded from: ../data/02qubits_05A.X.data.npy is: 9
Data loaded successfully from ../data/04qubits_05A.Y.data.npy
>>> Stat. >>> Size of set loaded from: ../data/04qubits_05A.Y.data.npy is: 9
Data loaded successfully from ../data/06qubits_05A.Y.data.npy
>>> Stat. >>> Size of set loaded from: ../data/06qubits_05A.Y.data.npy is: 225
Data loaded successfully from ../data/02qubits_05A.Y.data.npy
>>> Stat. >>> Size of set loaded from: ../data/02qubits_05A.Y.data.npy is: 9
Data loaded successfully from ../data/08qubits_05A.Y.data.npy
>>> Stat. >>> Size of set loaded from: ../data/08qubits_05A.Y.data.npy is: 225
Size of Xbig and Ybig: 468, 468
>> Data loaded okay
>> Size of each vector is: 598 with ham28 vec size 158
>>> Train the model with size  9
>>>>>>>>>>>>> Mean Squared Error: 732.6427612304688 :: Predictions: [-27.51541 -27.51541 -27.51541 -27.51541 -27.51541 -27.51541 -27.51541
 -27.51541 -27.51541 -27.51541 -27.51541 -27.51541 -27.51541 -27.51541
 -27.51541 -27.51541 -27.51541 -27.51541 -27.51541 -27.51541 -27.51541
 -27.51541 -27.51541 -27.51541 -27.51541 -27.51541 -27.51541 -27.51541
 -27.51541 -27.51541 -27.51541 -27.51541 -27.51541 -27.51541 -27.51541
 -27.51541 -27.51541 -27.51541 -27.51541 -27.51541 -27.51541 -27.51541
 -27.51541 -27.51541 -27.51541 -27.51541 -27.51541 -27.51541 -27.51541
 -27.51541 -27.51541 -27.51541 -27.51541 -27.51541 -27.51541 -27.51541
 -27.51541 -27.51541 -27.51541 -27.51541 -27.51541 -27.51541 -27.51541
 -27.51541 -27.51541 -27.51541 -27.51541 -27.51541 -27.51541 -27.51541
 -27.51541 -27.51541 -27.51541 -27.51541 -27.51541 -27.51541 -27.51541
 -27.51541 -27.51541 -27.51541 -27.51541 -27.51541 -27.51541 -27.51541
 -27.51541 -27.51541 -27.51541 -27.51541 -27.51541 -27.51541 -27.51541
 -27.51541 -27.51541 -27.51541] :: True Values: [np.float64(-0.27736972182193215), np.float64(-0.35306318597867414), np.float64(-2.240381970584433), np.float64(-0.34067792260946655), np.float64(-0.34802626300144573), np.float64(-0.37399665507721275), np.float64(-0.26959037548255127), np.float64(-0.2940193033868975), np.float64(-0.28153648969054246), np.float64(-0.27403477494789014), np.float64(-0.2823107255283967), np.float64(-0.3490222302692259), np.float64(-0.34522385259746385), np.float64(-0.2922494516171776), np.float64(-0.34426175863927555), np.float64(-0.2634361192056986), np.float64(-0.3284032312297771), np.float64(-0.3452692375036546), np.float64(-0.792611693074909), np.float64(-0.2797747705770556), np.float64(-0.2867673556423646), np.float64(-0.30322353756615905), np.float64(-0.3578213261802256), np.float64(-0.28895834626315475), np.float64(-0.25013690532120003), np.float64(-0.31600880730773245), np.float64(-0.34569530984636215), np.float64(-1.4347518742086465), np.float64(-0.4803341732955741), np.float64(-0.2938431426420781), np.float64(-0.3573902522828363), np.float64(-0.27891499511340817), np.float64(-0.2575409412023576), np.float64(-0.20917338632890714), np.float64(-0.3395935978425042), np.float64(-0.20787528116806842), np.float64(-0.2789277955300832), np.float64(-0.35595932231927235), np.float64(-0.2511780855847948), np.float64(-0.3599239887336513), np.float64(-0.3058819214299795), np.float64(-0.28683157624251215), np.float64(-0.2853615241100263), np.float64(-0.2587277725865946), np.float64(-0.2519417170440425), np.float64(-0.358995385793754), np.float64(-0.2848808973782924), np.float64(-0.28618368532506083), np.float64(-0.31207667889620627), np.float64(-0.2697452553410109), np.float64(-0.8876323362215769), np.float64(-0.26857992515926554), np.float64(-1.9330980550001378), np.float64(-0.29168236619811383), np.float64(-0.27982260800201986), np.float64(-0.32139418916792956), np.float64(-0.35185278517361207), np.float64(-0.36457671786292545), np.float64(-0.32295893090868877), np.float64(-0.8956371276855699), np.float64(-0.2995437807591365), np.float64(-0.2927354077758928), np.float64(-3.19079570893785), np.float64(-0.22184283224358284), np.float64(-0.29272185788889415), np.float64(-0.3407786400147354), np.float64(-0.293151525025574), np.float64(-0.3392431077575353), np.float64(-0.33886535692561615), np.float64(-0.27323211552587867), np.float64(-0.32587628299939947), np.float64(-0.33622370923953004), np.float64(-0.3395853112713748), np.float64(-0.2831262790086914), np.float64(-0.2938481382743163), np.float64(-0.29977460335902884), np.float64(-0.28041129264935816), np.float64(-0.3227909756984899), np.float64(-0.34737289921035175), np.float64(-0.6071131189337212), np.float64(-0.24793300906916302), np.float64(-0.28771874621546506), np.float64(-0.3014658437309368), np.float64(-0.3422257590728384), np.float64(-0.2949705232824317), np.float64(-0.3350414111734365), np.float64(-3.2315374519957283), np.float64(-0.2884178617895731), np.float64(-1.578534830005607), np.float64(-0.32741150599285573), np.float64(-0.28835170973175756), np.float64(-0.35689159910090124), np.float64(-0.34097308575374463), np.float64(-0.29473995280718174)]
>>>>>>>>>>>>> Mean ABS Error: 27.06240463256836
>>>>>>>>>>>>> Mean Squared Error: 722.339111328125 :: Predictions: [-27.51541 -27.51541 -27.51541 -27.51541 -27.51541 -27.51541 -27.51541
 -27.51541 -27.51541 -27.51541 -27.51541 -27.51541 -27.51541 -27.51541
 -27.51541 -27.51541 -27.51541 -27.51541 -27.51541 -27.51541 -27.51541
 -27.51541 -27.51541 -27.51541 -27.51541 -27.51541 -27.51541 -27.51541
 -27.51541 -27.51541 -27.51541 -27.51541 -27.51541 -27.51541 -27.51541
 -27.51541 -27.51541 -27.51541 -27.51541 -27.51541 -27.51541 -27.51541
 -27.51541 -27.51541 -27.51541 -27.51541 -27.51541 -27.51541 -27.51541
 -27.51541 -27.51541 -27.51541] :: True Values: [np.float64(-28.683086710655314), np.float64(-0.3572420073829669), np.float64(-0.3546108433644162), np.float64(-0.3479770325278414), np.float64(-0.34389659184975485), np.float64(-0.2124985480850643), np.float64(-0.34420519031255237), np.float64(-0.34636287905110924), np.float64(-0.35339058298224024), np.float64(-0.32578536790183266), np.float64(-0.21837040923144854), np.float64(-0.3458554654568524), np.float64(-0.3459020421290991), np.float64(-0.34494277580834043), np.float64(-1.2688770612114064), np.float64(-0.35744578765012197), np.float64(-0.35161966865264516), np.float64(-0.20580386994958866), np.float64(-0.3575334765322117), np.float64(-0.3567712718511922), np.float64(-0.34382963560632734), np.float64(-0.37090690938602017), np.float64(-0.3299776085085959), np.float64(-0.49793868470000635), np.float64(-0.33489277187704813), np.float64(-1.138592517822652), np.float64(-0.8115519802840252), np.float64(-0.29867190588265446), np.float64(-0.2838262868421753), np.float64(-0.43854564673692825), np.float64(-0.27284688796139334), np.float64(-0.25219960080856596), np.float64(-0.28591425031463785), np.float64(-0.27309249580920475), np.float64(-0.25144781770730307), np.float64(-0.295298373362151), np.float64(-0.28476747496770505), np.float64(-0.24887906004388446), np.float64(-0.27104513254481505), np.float64(-1.4672595364909427), np.float64(-0.2907239303837679), np.float64(-0.24963048838413016), np.float64(-0.26498069977430244), np.float64(-0.2891418405054605), np.float64(-0.28604385469019095), np.float64(-0.3267762420080949), np.float64(-0.2533311082646857), np.float64(-0.2828946230939801), np.float64(-0.30252925542170234), np.float64(-0.2899491610127196), np.float64(-0.2858376706779248), np.float64(-0.28796949349556444)]
>>>>>>>>>>>>> Mean ABS Error: 26.637561798095703
>>> Stat. >>> Size of TRAIN set: 374
>>> Stat. >>> Size of TEST set: 94
>>> Stat. >>> Size of EXTRA TEST set: 52
>> End Training. Max Size is 
598
total 288
-rw-r--r-- 1 root root    128 Jun 25 11:57 02qubits_05.X.data.npy
-rw-r--r-- 1 root root    128 Jun 25 11:57 02qubits_05.Y.data.npy
-rw-r--r-- 1 root root  47408 Jun 25 11:57 04qubits_05.X.data.npy
-rw-r--r-- 1 root root    208 Jun 25 11:57 04qubits_05.Y.data.npy
-rw-r--r-- 1 root root  86128 Jun 25 11:57 06qubits_05.X.data.npy
-rw-r--r-- 1 root root   2128 Jun 25 11:57 06qubits_05.Y.data.npy
-rw-r--r-- 1 root root 134128 Jun 25 11:57 08qubits_05.X.data.npy
-rw-r--r-- 1 root root   2128 Jun 25 11:57 08qubits_05.Y.data.npy
>> Running kcl_adapt_vqe_stage_2.py
>> Start Stage 2
>> Read ham ../hamiltonian/, 28qubits_01.data
28qubits_01
>> Start processing: 28qubits_01.data with qubits 28
>>>> adding ham of size 98700
>> Start Training
Load data from folder ../data/
Data loaded successfully from ../data/02qubits_05.X.data.npy
>>> Stat. >>> Size of set loaded from: ../data/02qubits_05.X.data.npy is: 0
Data loaded successfully from ../data/06qubits_05.X.data.npy
>>> Stat. >>> Size of set loaded from: ../data/06qubits_05.X.data.npy is: 225
Data loaded successfully from ../data/08qubits_05.X.data.npy
>>> Stat. >>> Size of set loaded from: ../data/08qubits_05.X.data.npy is: 225
Data loaded successfully from ../data/04qubits_05.X.data.npy
>>> Stat. >>> Size of set loaded from: ../data/04qubits_05.X.data.npy is: 9
Data loaded successfully from ../data/06qubits_05.Y.data.npy
>>> Stat. >>> Size of set loaded from: ../data/06qubits_05.Y.data.npy is: 225
Data loaded successfully from ../data/08qubits_05.Y.data.npy
>>> Stat. >>> Size of set loaded from: ../data/08qubits_05.Y.data.npy is: 225
Data loaded successfully from ../data/04qubits_05.Y.data.npy
>>> Stat. >>> Size of set loaded from: ../data/04qubits_05.Y.data.npy is: 9
Data loaded successfully from ../data/02qubits_05.Y.data.npy
>>> Stat. >>> Size of set loaded from: ../data/02qubits_05.Y.data.npy is: 0
Size of Xbig and Ybig: 459, 459
>> Data loaded okay
>> Size of each vector is: 604 with ham28 vec size 158
>>> Train the model with size  9
>>>>>>>>>>>>> Mean Squared Error: 0.06566351652145386 :: Predictions: [-0.69486463 -0.62099147 -0.69222534 -0.6243621  -0.63472193 -0.6853166
 -1.0229478  -0.621587   -0.8945026  -0.6936969  -0.62184125 -0.5491872
 -0.627981   -0.62097186 -0.6207306  -0.60641    -0.62079906 -0.6939242
 -0.6949743  -0.6242982  -0.6902838  -0.6942592  -0.68445283 -0.6949539
 -0.6209323  -3.8698044  -0.6212845  -0.6941513  -0.6204986  -0.60365343
 -0.67506033 -0.62332606 -0.63198763 -1.7377844  -0.938341   -0.620996
 -0.6071806  -0.702447   -0.6847727  -0.7008076  -0.6210416  -0.7160663
 -0.66882163 -0.6959954  -3.8698044  -0.6922914  -0.6132871  -0.62890035
 -1.0915811  -1.2971685  -0.69469583 -0.8650357  -0.79621017 -0.6198123
 -0.6940887  -0.61029994 -0.62022716 -0.72398984 -0.8022973  -0.6962699
 -0.62138265 -0.6212472  -0.69470227 -0.62008506 -0.69568694 -0.682501
 -0.6306662  -0.6239764  -0.6207862  -0.66036373 -0.8484502  -0.62175333
 -0.62477314 -0.6919839  -0.73389375 -0.6000681  -0.694379   -1.0700439
 -0.6942592  -0.6934448  -0.6945286  -0.6286163  -0.68166775 -0.69438756
 -0.6846374  -0.67485076 -0.62221295 -0.61253816 -0.88858026 -0.6207831
 -0.69008553 -0.6960579 ] :: True Values: [np.float64(-0.695939025880892), np.float64(-0.6175164811952343), np.float64(-0.695939025880892), np.float64(-0.6175164811952344), np.float64(-0.6175164811952343), np.float64(-0.695939025880892), np.float64(-0.695939025880892), np.float64(-0.6175164811952344), np.float64(-0.695939025880892), np.float64(-0.695939025880892), np.float64(-0.6175164811952344), np.float64(-0.6175164811952343), np.float64(-0.6175164811952344), np.float64(-0.6175164811952344), np.float64(-0.6175164811952343), np.float64(-0.6175164811952343), np.float64(-0.6175164811952343), np.float64(-0.695939025880892), np.float64(-0.695939025880892), np.float64(-0.6175164811952344), np.float64(-0.695939025880892), np.float64(-0.695939025880892), np.float64(-0.695939025880892), np.float64(-0.695939025880892), np.float64(-0.6175164811952343), np.float64(-5.251192387396456), np.float64(-0.6175164811952343), np.float64(-0.695939025880892), np.float64(-0.6175164811952344), np.float64(-0.6175164811952344), np.float64(-0.6175164811952343), np.float64(-0.6175164811952344), np.float64(-0.6175164811952343), np.float64(-0.695939025880892), np.float64(-0.6175164811952343), np.float64(-0.6175164811952343), np.float64(-0.6175164811952343), np.float64(-0.695939025880892), np.float64(-0.695939025880892), np.float64(-0.695939025880892), np.float64(-0.6175164811952344), np.float64(-0.695939025880892), np.float64(-0.695939025880892), np.float64(-0.695939025880892), np.float64(-5.251192387396458), np.float64(-0.695939025880892), np.float64(-0.6175164811952343), np.float64(-0.6175164811952343), np.float64(-0.695939025880892), np.float64(-0.6175164811952344), np.float64(-0.695939025880892), np.float64(-0.695939025880892), np.float64(-0.695939025880892), np.float64(-0.6175164811952343), np.float64(-0.695939025880892), np.float64(-0.6175164811952343), np.float64(-0.6175164811952343), np.float64(-0.695939025880892), np.float64(-0.695939025880892), np.float64(-0.695939025880892), np.float64(-0.6175164811952343), np.float64(-0.6175164811952343), np.float64(-0.695939025880892), np.float64(-0.6175164811952343), np.float64(-0.695939025880892), np.float64(-0.6175164811952344), np.float64(-0.6175164811952344), np.float64(-0.6175164811952343), np.float64(-0.6175164811952344), np.float64(-0.6175164811952344), np.float64(-0.695939025880892), np.float64(-0.6175164811952343), np.float64(-0.6175164811952344), np.float64(-0.695939025880892), np.float64(-0.695939025880892), np.float64(-0.6175164811952344), np.float64(-0.695939025880892), np.float64(-0.695939025880892), np.float64(-0.695939025880892), np.float64(-0.695939025880892), np.float64(-0.695939025880892), np.float64(-0.6175164811952344), np.float64(-0.695939025880892), np.float64(-0.695939025880892), np.float64(-0.695939025880892), np.float64(-0.695939025880892), np.float64(-0.6175164811952343), np.float64(-0.6175164811952343), np.float64(-0.695939025880892), np.float64(-0.6175164811952344), np.float64(-0.695939025880892), np.float64(-0.695939025880892)]
>>>>>>>>>>>>> Mean ABS Error: 0.08213764429092407
>>>>>>>>>>>>> Mean Squared Error: 0.04934525117278099 :: Predictions: [-0.6927328  -0.6930152  -0.6933431  -0.6942038  -0.7336743  -1.2783968
 -0.6939425  -0.6951648  -0.69583076 -0.69646615 -0.6944386  -0.6946403
 -0.69441736 -0.8658571  -0.69243795 -0.6875726  -0.68390685 -0.704471
 -0.641113   -0.69445986 -0.6838034  -0.7068113  -0.6962464  -0.68980265
 -1.0970107  -0.62056214 -0.6188864  -0.6165956  -0.6209882  -0.62098885
 -0.62101644 -0.6223847  -0.619822   -0.6212535  -0.6200077  -0.59033924
 -0.62098104 -0.63098735 -0.6203727  -0.6334986  -0.57577306 -0.6206532
 -0.6215927  -0.62069905 -0.6179765  -0.6201034  -0.6215824  -0.62432426
 -0.6225422  -0.6204882  -3.8442922 ] :: True Values: [np.float64(-0.695939025880892), np.float64(-0.695939025880892), np.float64(-0.695939025880892), np.float64(-0.695939025880892), np.float64(-0.695939025880892), np.float64(-0.695939025880892), np.float64(-0.695939025880892), np.float64(-0.695939025880892), np.float64(-0.695939025880892), np.float64(-0.695939025880892), np.float64(-0.695939025880892), np.float64(-0.695939025880892), np.float64(-0.695939025880892), np.float64(-0.695939025880892), np.float64(-0.695939025880892), np.float64(-0.695939025880892), np.float64(-0.695939025880892), np.float64(-0.695939025880892), np.float64(-0.695939025880892), np.float64(-0.695939025880892), np.float64(-0.695939025880892), np.float64(-0.695939025880892), np.float64(-0.695939025880892), np.float64(-0.695939025880892), np.float64(-0.695939025880892), np.float64(-0.6175164811952343), np.float64(-0.6175164811952343), np.float64(-0.6175164811952344), np.float64(-0.6175164811952344), np.float64(-0.6175164811952344), np.float64(-0.6175164811952344), np.float64(-0.6175164811952344), np.float64(-0.6175164811952343), np.float64(-0.6175164811952344), np.float64(-0.6175164811952343), np.float64(-0.6175164811952344), np.float64(-0.6175164811952344), np.float64(-0.6175164811952343), np.float64(-0.6175164811952343), np.float64(-0.6175164811952343), np.float64(-0.6175164811952344), np.float64(-0.6175164811952343), np.float64(-0.6175164811952343), np.float64(-0.6175164811952343), np.float64(-0.6175164811952344), np.float64(-0.6175164811952343), np.float64(-0.6175164811952343), np.float64(-0.6175164811952344), np.float64(-0.6175164811952343), np.float64(-0.6175164811952344), np.float64(-5.251192387396458)]
>>>>>>>>>>>>> Mean ABS Error: 0.056875504553318024
>>> Stat. >>> Size of TRAIN set: 367
>>> Stat. >>> Size of TEST set: 92
>>> Stat. >>> Size of EXTRA TEST set: 51
>> End Training. Max Size is 
604
>> Phase 2 complete.
```

#### Full-Evaluation

This script mines data from **all** small systems using all the data we have in the Zenodo record using a GPU.
```
cd Artifact_Experiments
chmod 777 phase_2_full.sh
./phase_2_full.sh
```
If you try it with CPU = 1, you are very likely to get Out-of-Memory "```./phase_2_full.sh: line 26:  6350 Killed                  python3 kcl_QCELS_stage_2.py```", 
which is expected. This script is for GPU use only.

With GPU, you shall change
```
cpu=1 # >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>> Change when using a GPU
```
to
```
cpu=0 # >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>> Change when using a GPU
```
in both wrapper scripts: ```kcl_QCELS_stage_2.py``` and ```kcl_adapt_vqe_stage_2.py```.





### 3.5 Phase 3 - Model Deployment

Once you have the models from Phase 2, you can test them by copying them to the model folder. However, unless trained via GPU, these are likely to perform extremely poorly.
You can use our pre-trained models, which are already in the model folder.

As before, we have a wrapper script for each QE implementation:
```
python3 kcl_QCELS_stage_3.py
python3 kcl_adapt_vqe_stage_3.py
```

For phase 3, these parts are relevant:
```
AccelerQ-main
├── src
     ├── kcl_QCELS_stage_3.py, kcl_adapt_vqe_stage_3.py     # optimisation wrappers
     ├── kcl_tests_qcels.py                                 # QCELS evaluation oracles (test_static_qcels, test_semi_dynamic_qcels)
     ├── kcl_tests_adapt_vqe.py                             # adapt-VQE evaluation oracles (test_static_adapt, test_semi_dynamic_adapt)
     ├── kcl_opt_xgb.py                                     # hyperparameter optimisation logic (opt_hyperparams)
     ├── kcl_util.py                                        # shared utilities (process_file, ham_to_vector, load_model, print_to_file)
     ├── kcl_util_qcels.py                                  # QCELS parameter generation (generate_hyper_params_qcels)
     ├── kcl_util_adapt_vqe.py                              # adapt-VQE parameter generation (generate_hyper_params_avqe)
```

#### Partial Evaluation **fit for a laptop**

Some systems can take longer, we picked 3 that we know fit for CPU.
```
cd Artifact_Experiments
cp ../models/* ../src/   # See note below.
chmod 777 phase_3_short.sh
./phase_3_short.sh
```
This script on X86 with 8 GB RAM ran in our Docker for 8 minutes. Data will be written into ```src``` folder:

**NOTE:** You can use the models you trained, but you will need to update the max size of the record, which is stated at the end of phase 2, per model trained. In the examples here, this was:
```
>> End Training. Max Size is 
598
```
and
```
>> End Training. Max Size is 
604
```

#### Full-Evaluation

This script runs the full optimisation for all systems and both QE implementations. This can take around 1 week to finish on a CPU.
```
cd Artifact_Experiments
chmod 777 phase_3_full.sh
./phase_3_full.sh
```

### 3.6 Figure 5 - Hyperparameters' value distribution

We checked 
