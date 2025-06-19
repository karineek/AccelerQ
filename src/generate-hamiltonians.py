import openfermion as of
from openfermion import FermionOperator as FO
from openfermion import load_operator
from openfermion.transforms import jordan_wigner

import random
import os


def generate_single_hamiltonian(number_orbitals: int):
    """
    Generates a single Hamiltonian with the specified number of orbitals
        Parameters:
            number_orbitals: number of orbital terms in the hamiltonian
        Return:
            generalised hamiltonian with randomly generated hopping and Coulomb terms.
    """
    H = 0.0
    for i in range(number_orbitals):
        for j in range(number_orbitals):
            t_term = -random.uniform(0.0, 2.0) * FO(
                "[{0}^ {1}]".format(i, j)
            )  # hopping energy
            H += t_term
            for k in range(number_orbitals):
                for l in range(number_orbitals):
                    U_term = random.uniform(0.0, 2.0) * FO(
                        "[{0}^ {1} {2}^ {3}]".format(i, j, k, l)
                    )  # Coulomb energy
                    H += U_term
    return H


# stringify single and double digit numbers for filenames
def format_number_str(number: int):
    return number if number > 9 else f"0{number}"


# save the input hamiltonian in 'hamiltonian' folder in '.data' format
def save_hamiltonian(hamiltonian: FO, number_of_qubits: int, seed: int):
    seed_str = format_number_str(seed)
    number_of_qubits_str = format_number_str(number_of_qubits)
    file_name = f"{number_of_qubits_str}qubits_{seed_str}.data"
    print(f"Saving file {file_name}")
    of.utils.save_operator(
        hamiltonian,
        file_name=file_name,
        data_directory="../hamiltonian",
        allow_overwrite=False,
        plain_text=False,
    )


# search through 'hamiltonian' directory for the next available seed
def check_available_seed():
    # assign directory
    directory = "../hamiltonian"
    max_seed = 0

    for filename in os.listdir(directory):
        f = os.path.join(directory, filename)
        if os.path.isfile(f) and ".data" in f:
            seed = int(filename[9:11])
            if seed > max_seed:
                max_seed = seed

    return max_seed + 1


def generate_hamiltonians(number_of_hamiltonians: int = 10):
    """
    Generates and saves Hamiltonians to the 'hamiltonian' folder
        Parameters:
            number_of_hamiltonians: number of hamiltonians to generate
    """
    min_qubits = 4
    seed = check_available_seed()
    for i in range(number_of_hamiltonians):
        number_of_qubits = min_qubits + i
        hamiltonian = generate_single_hamiltonian(number_of_qubits)
        print(f"Generating Hamiltonian with {number_of_qubits} number of qubits.")
        save_hamiltonian(hamiltonian, number_of_qubits, seed)


# remove hamiltonian files from 'hamiltonian' folder with the specified input 'seed'
def remove_hamiltonians(seed: int):
    # assign directory
    directory = "../hamiltonian"
    seed_str = format_number_str(seed)

    for filename in os.listdir(directory):
        f = os.path.join(directory, filename)
        if os.path.isfile(f) and f"{seed_str}.data" in f:
            try:
                os.remove(f)
                print(f"File {filename} removed.")
            except OSError as e:
                print("Failed with:", e.strerror)
                print("Error code:", e.code)


if __name__ == "__main__":
    # generates and saves the specified number of Hamiltonians
    number_of_hamiltonians = 10
    generate_hamiltonians(number_of_hamiltonians)
