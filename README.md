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


## ADPT-QSCI

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

## QCELS

We wrote a version of the algorithm. See QCELS/QCELS_answer.py. This code is only for reference.

## Python Packages

Please take a look at the dependencies in the requirements.txt file.

## Setup the rest of the Experimental environment

You might need to create a swapfile to run the ADPT-QSCI and QCELS with 20+ qubits.
```
./AccelerQ/scripts/0-swap-setup.sh <YOUR-HOME-DIR>
```

### Hardware Specifications

- Architectures: x86, ARM
