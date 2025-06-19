# ------ ADDED IMPORT ------- TODO: Check if we can keep them
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../src')))
import hashlib         # STAB - REMOVE
import json            # STAB - REMOVE
import sys
import traceback
from random import randint
from itertools import combinations
from typing import Any, Optional, Union, Tuple, List, Sequence, Mapping

from openfermion import QubitOperator, jordan_wigner
from openfermion.transforms import jordan_wigner

from qiskit import quantum_info

from first_answer import RunAlgorithm

from quri_parts.algo.ansatz import SymmetryPreservingReal
from quri_parts.algo.optimizer import SPSA, OptimizerStatus
from quri_parts.circuit import LinearMappedUnboundParametricQuantumCircuit
from quri_parts.circuit.transpile import RZSetTranspiler
from quri_parts.core.estimator.gradient import parameter_shift_gradient_estimates
from quri_parts.core.measurement import bitwise_commuting_pauli_measurement, CachedMeasurementFactory
from quri_parts.core.operator import (
    pauli_label,
    Operator,
    PauliLabel,
    pauli_product,
    PAULI_IDENTITY,
)
from quri_parts.core.operator.representation import (
    BinarySymplecticVector,
    pauli_label_to_bsv,
    transition_amp_representation,
    transition_amp_comp_basis,
)
from quri_parts.core.sampling.shots_allocator import create_equipartition_shots_allocator
from quri_parts.core.state import ComputationalBasisState, ParametricCircuitQuantumState
from quri_parts.openfermion.operator import operator_from_openfermion_op
from quri_parts.qulacs.sampler import create_qulacs_vector_sampler

sys.path.append("../")
from utils.challenge_2024 import ChallengeSampling, ExceededError, problem_hamiltonian

# Imports for ML part
import numpy as np
import stopit
from datetime import datetime
import random
import copy
import xgboost as xgb
os.environ['KMP_DUPLICATE_LIB_OK']='True'

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, recall_score
from sklearn.pipeline import make_pipeline
from sklearn.metrics import mean_squared_error

os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

# A bit of code to make it all installed and resolve conflicts
challenge_sampling = ChallengeSampling()
print("DONE")
