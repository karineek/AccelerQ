# AccelerQ

Evaluation was done with two QE solver implementation. While the QCELS implementation is supplied with this Git Repository, you will have to 
pull the ADPT-QSCI code (the part that required no changes) from the original Git Repository.

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
