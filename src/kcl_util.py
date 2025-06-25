
"""
This script is 

This file is part of the AccelerQ Project.
(2025) King's College London. CC BY 4.0.
- You must give appropriate credit, provide a link to the license, and indicate if changes 
  were made. You may do so in any reasonable manner, but not in any way that suggests 
  the licensor endorses you or your use.

"""

import os
import sys
import glob
import numpy as np
from datetime import datetime
#import joblib
#import pickle

from openfermion import FermionOperator as FO
from openfermion import load_operator
sys.path.append("../")

from utils.challenge_2024 import problem_hamiltonian # Needed for process_file - can remove if discard it.




#===========================================
# FUNCTIONS for Reading and Writing to Files
#===========================================


def save_to_binary(file_name, data):
    """
    Save numerical data to a binary file using NumPy.

    Parameters:
    file_name (str): The name of the file to save the data.
    data (numpy.ndarray or list): The data to save.
    """
    # Convert list to NumPy array if necessary
    if isinstance(data, list):
        data = np.array(data)
    np.save(file_name, data)
    print(f"Data written successfully to {file_name}")

def load_from_binary(file_name):
    """
    Load numerical data from a binary file using NumPy.

    Parameters:
    file_name (str): The name of the file to load the data from.

    Returns:
    numpy.ndarray: The loaded data.
    """
    data = np.load(file_name)
    print(f"Data loaded successfully from {file_name}")
    return data

def simple_load_ham(folder, file):
    """
    Load the Hamiltonian using the provided folder and file.

    Parameters:
    folder (str): The path to the folder containing the file.
    file (str): The name of the file to load.

    Returns:
    The loaded Hamiltonian.
    """
    print(">> Load: " + folder + "," + file)
    ham = load_operator(
        file_name=file,
        data_directory=folder,
        plain_text=False
    )
    return ham

def process_file(folder_path, file_name):
    """
    Reads the Hamiltonian file and processes it.

    Parameters:
    folder_path (str): The path to the folder containing the file.
    file_name (str): The name of the file to process.

    Returns:
    tuple: A tuple containing the first two letters of the file name and the Hamiltonian.
    """
    # Construct the file path
    file_path = os.path.join(folder_path, file_name)
    # Read the problem Hamiltonian
    postfix = file_name[-7:]
    ham = problem_hamiltonian(file_name[:2], postfix[:2], folder_path)
    # Return a tuple containing the first two letters of the file name and its content
    return file_name[:2], ham

def print_to_file(message):
    """
    Append a message to a log file with timestamp.

    Parameters:
    message (str): The message to append to the log file.
    """
    with open("logger.txt", 'a') as f:
        current_time = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        f.write(f"{current_time} - {message}\n")

def load_and_split_data(files):
    """
    Load population from files and splits to 90-10.
    """
    # Initialize empty lists to collect the split data
    set90 = []
    set10 = []

    for file in files:
        # Load data from file
        data = load_from_binary(file)

        # Split the data into 80% and 20%
        split_index = int(len(data) * 0.9)
        data_90 = data[:split_index]
        data_10 = data[split_index:]

        # Add the split data to the respective lists
        for x in data_90:
            set90.append(x)
        for x in data_10:
            set10.append(x)

        # Add information for the paper
        print (">>> Stat. >>> Size of set loaded from: " + file + " is: " + str(len(data_90)))

    return set90, set10

def load_data_set(folder):
    """
    Load all *.X.*.data.npy and *.Y.*.data.npy files in the specified
    folder, split each file's data into 90% and 10% portions,
    and add the splits to four arrays.

    Parameters:
    folder (str): The path to the folder containing the .npy files.

    Returns:
    tuple: Two numpy arrays, X's and Y's
    """
    # Get list of all *.X.*.data.npy and *.Y.*.data.npy files in the folder
    file_patternX = os.path.join(folder, "*.X.data.npy")
    filesX = glob.glob(file_patternX)
    file_patternY = os.path.join(folder, "*.Y.data.npy")
    filesY = glob.glob(file_patternY)

    # Load and split
    Xbig, Xsmall = load_and_split_data(filesX)
    Ybig, Ysmall = load_and_split_data(filesY)

    # Test size is okay
    if len(Xbig) != len(Ybig):
        print("size of X: " + str(len(Xbig)))
        print("size of Y: " + str(len(Ybig)))
        raise ValueError("Size of X and Y must be equal")

    if len(Xsmall) != len(Ysmall):
        print("size of X (small): " + str(len(Xsmall)))
        print("size of Y (small): " + str(len(Ysmall)))
        raise ValueError("Size of X and Y (small) must be equal")

    # Return data set
    print ("Size of Xbig and Ybig: " + str(len(Xbig)) + ", " + str(len(Ybig)))
    return Xbig, Ybig, Xsmall, Ysmall

def save_model(model_file, model):
    model.save_model(model_file)

def load_model(model_file, type):
    model = type()
    model.load_model(model_file)
    return model



#==========================================
# FUNCTIONS for Quantum to ML represntation
#==========================================


def flatten(nested_list):
    """
    Flatten a nested list or tuple.

    Parameters:
    nested_list (list or tuple): The nested list or tuple to flatten.

    Returns:
    list: A flattened list with all values converted to floats.
    """
    flat_list = []
    for item in nested_list:
        if isinstance(item, (list, tuple)):
            flat_list.extend(flatten(item))
        else:
            flat_list.append(float(item))
    return flat_list

def ham_to_vector(ham, reduce=1.0):
    """
    Convert Hamiltonian to a flat vector.

    Parameters:
    ham (FermionOperator): The Hamiltonian to convert.

    Returns:
    numpy.ndarray: The flat vector representation of the Hamiltonian.
    """
    vector = []

    if isinstance(ham, FO):
        # Extract the terms and coefficients
        terms = ham.terms
        vector = flatten(flatten(terms))
        vector = [element for element in vector if not (-(reduce) <= element <= reduce)]
        # pick top 1000 - check on ABS
        sorted_vector = np.sort(vector)

        # Iterate over the terms and their coefficients
        for term, coefficient in terms.items():
            vector.append(float(coefficient.real))

        # Convert to a numpy array
        vector = [element for element in vector if not (-0.2 <= element <= 0.2)]
        vector = np.array(flatten(vector), order='C', dtype=float)

        # Print to log
        print(">>>> adding ham of size " + str(len(terms)))

    else:
        typo = type(ham).__name__
        raise TypeError("Expected ham to be a FermionOperator, but got {}".format(typo))

    # Return the flat vector
    return vector

def vec_to_fixed_size_vec(size, vec_in):
    """
    Ensure all Hamiltonians are represented using the same number of bytes.

    Parameters:
    size (int): The desired size of the output vector.
    vec_in (list or numpy.ndarray): The input vector to be resized.

    Returns:
    numpy.ndarray: The resized vector, padded with zeros if necessary.

    Raises:
    ValueError: If the input vector is larger than the desired size.
    """
    vec = np.array(vec_in)

    # Pad with zeros if vec is shorter than size
    if len(vec) < size:
        return np.pad(vec, (0, size - len(vec)), 'constant')

    # Return vec as is if it is already of the correct size
    elif len(vec) > size:
        raise ValueError("Size of X - padding failed. Vec larger than max size: " + str(size))

    else:
        return vec
