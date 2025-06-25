An implementation of the algorithm [2] is provided in [QCELS/QCELS_answer.py](https://github.com/karineek/AccelerQ/blob/main/QCELS/QCELS_answer.py), copied as is from [1].

**Note on this implementation:** it follows the Hamiltonian model. That is, rather than using discrete gates, the Hamiltonian model computes by evolving the quantum system continuously in time under a time-dependent Hermitian matrix $$H(t)$$. This evolution is governed by **Schrödinger’s equation**:

$$
i \hbar \frac{d}{dt} |\psi(t)\rangle = H(t) |\psi(t)\rangle
$$

[1] Connorpl. Accessed: July 4, 2024. QCELS_for_QAGC. [https://github.com/Connorpl/QCELS_for_QAGC].

[2] Zhiyan Ding and Lin Lin. 2023. Even Shorter Quantum Circuit for Phase Estimation on Early Fault-Tolerant Quantum Computers with Applications to
Ground-State Energy Estimation. PRX Quantum 4 (May 2023), 020331. Issue 2. [https://doi.org/10.1103/PRXQuantum.4.020331]

**This is not our code, and we do not claim ownership. Please refer to the original GitHub Repository if you wish to reuse this code.**
