
"""
This script processes Hamiltonian systems and extracts size before and after truncation 
under different parameters. It iterates over a set of Hamiltonian datasets. The script 
outputs detailed numerical results for the size of systems (Hamiltonians) graph/table 
in the evaluation.

This file is part of the AccelerQ Project.
(2025) King's College London. CC BY 4.0.
- You must give appropriate credit, provide a link to the license, and indicate if changes 
  were made. You may do so in any reasonable manner, but not in any way that suggests 
  the licensor endorses you or your use.

"""

import sys
import os
import numpy as np
from kcl_util import process_file, ham_to_vector
from kcl_util_qcels import generate_hyper_params_qcels
from kcl_tests_qcels import print_1

def main():
    # Set up
    np.set_printoptions(precision=17)
    folder_path = "../hamiltonian/"

    # List of (num_qubits, seed) pairs to use in function calls
    num_qubits_seed_pairs = [
         (20, 0), (20, 1), (20, 2), (20, 3), (20, 4), (20, 5),
         (24, 5), (24, 6), (24, 7), (24, 8), (24, 9),
         (28, 0), (28, 1), (28, 2), (28, 3), (28, 4)
    ]

    if len(sys.argv) == 3:
        is_OPT = int(sys.argv[1]) != 0
        is_TEST = int(sys.argv[2]) != 0
        print(f">> is_OPT  = {is_OPT}")
        print(f">> is_TEST = {is_TEST}")
    else:
        print("Usage: python3 size_stat_qcels.py <is_OPT> <is_TEST>")
        sys.exit(1)
      
    for num_qubits, seed in num_qubits_seed_pairs:
        n_qubits = num_qubits
        if is_TEST:
            if n_qubits == 20:
                if (seed == 0):
                    res_opt = np.array(
                          [
                           2.00000000000000000e+01, 1.00000000000000000e+01, 2.03553255544773021e-01,
                           9.00000000000000000e+00, 9.80000000000000000e+01, 8.40990934260731071e-03,
                           5.03579034261508962e-01 ] # 20,0, n_z = 9

                        , dtype=float)
                elif (seed == 1):
                    res_opt = np.array(
                          [2.00000000000000000e+01, 1.00000000000000000e+01, 2.02095003116297856e-01,
                           1.85000000000000000e+01, 1.46000000000000000e+02, 8.34979425264609568e-03,
                           5.17756802571591512e-01 ] # 20,1, new, n_z = 19
                        , dtype=float)

                elif (seed == 2):
                    res_opt = np.array(
                          [20.0, 10.0, 0.20833096205618162, 10.0, 116.0, 0.008390067341719952, 0.5240496926349418] # 20,2, new n_z=10
                        , dtype=float)

                elif (seed == 3):
                    res_opt = np.array(
                          [
                             2.00000000000000000e+01, 1.00000000000000000e+01, 2.01424723474243950e-01,
                             8.00000000000000000e+00, 1.74000000000000000e+02, 8.33577013824858895e-03,
                             5.03754950047298200e-01 ] # 20,3, n_z = 8
                        , dtype=float)

                elif (seed == 4):
                    res_opt = np.array(
                          [2.00000000000000000e+01, 1.00000000000000000e+01, 2.00190554895460671e-01,
                           2.40000000000000000e+01, 1.67500000000000000e+02, 8.41137406817806139e-03,
                           5.45927804638917769e-01 ] # 20,4, new n_z = 24
                        , dtype=float)

                elif (seed == 5):
                    res_opt = np.array(
                          [
                          2.00000000000000000e+01, 1.00000000000000000e+01, 2.17675856508570603e-01,
                          2.00000000000000000e+01, 6.70000000000000000e+01, 9.73263144301033702e-03,
                          5.07903285275227545e-01 ] # 20,5, n_z=20
                        , dtype=float)

                else:
                    print ("EERROR n_qubits is : " + str(n_qubits) + " with seed" + str(seed))
                    return 0

            elif n_qubits == 24:
                if (seed == 5):
                    res_opt = np.array(
                          [
                          2.40000000000000000e+01, 1.20000000000000000e+01, 1.99802185604540827e-01,
                          1.70000000000000000e+01, 7.90000000000000000e+01, 9.95336063227277197e-03,
                          5.77057365918694831e-01 ] # 24,5, n_z=17
                        , dtype=float)

                elif (seed == 6):
                    res_opt = np.array(
                          [
                          2.40000000000000000e+01, 1.20000000000000000e+01, 2.02208370095536688e-01,
                          1.70000000000000000e+01, 7.50000000000000000e+01, 9.81354737396960934e-03,
                          5.92811924761539988e-01 ] # 24,6, n_z=17
                        , dtype=float)

                elif (seed == 7):
                    res_opt = np.array(
                          [
                          2.40000000000000000e+01, 1.20000000000000000e+01, 1.96109431581716925e-01,
                          2.40000000000000000e+01, 6.75000000000000000e+01, 8.40633981198942067e-03,
                          6.02858839442908190e-01 ] # 24,7, n_z=24
                        , dtype=float)

                elif (seed == 8):
                    res_opt = np.array(
                          [
                          2.40000000000000000e+01, 1.20000000000000000e+01, 2.03033308059204165e-01,
                          9.00000000000000000e+00, 7.30000000000000000e+01, 9.97384266029652938e-03,
                          5.52657866550458787e-01 ] # 24,8, n_z=9
                        , dtype=float)

                elif (seed == 9):
                    res_opt = np.array(
                          [
                          2.40000000000000000e+01, 1.20000000000000000e+01, 2.09114207651129175e-01,
                          1.60000000000000000e+01, 7.80000000000000000e+01, 8.34624269390677256e-03,
                          5.65200801771179151e-01 ] # 24,9, n_z=16
                       , dtype=float)

                else:
                    print ("EERROR n_qubits is : " + str(n_qubits) + " with seed" + str(seed))
                    return 0

            elif n_qubits == 28:
                if (seed == 0):
                    res_opt = np.array(
                          [2.8000000000000000e+01,1.4000000000000000e+01,7.0017442061948805e-03,
                          7.0000000000000000e+00,3.1000000000000000e+02,6.7115397362500061e-03,6.3276343991833195e-01] # 28,0
                       , dtype=float)

                elif (seed == 1):
                    res_opt = np.array(
                          [2.8000000000000000e+01,1.4000000000000000e+01,1.9971283221559202e-01,
                          2.3000000000000000e+01,3.0900000000000000e+02,8.3995728796683409e-03,5.5405304641464770e-01] # 28,1
                       , dtype=float)

                elif (seed == 2):
                    res_opt = np.array(
                          [ 2.80000000000000000e+01, 1.40000000000000000e+01, 1.97641575160143690e-01,
  7.00000000000000000e+00, 2.35000000000000000e+02, 8.38759959223706747e-03,
  5.15088244417049257e-01 ] # 28,2
                       , dtype=float)

                elif (seed == 3):
                    res_opt = np.array(
                          [ 2.80000000000000000e+01, 1.40000000000000000e+01, 1.99572630213638663e-01,
  8.00000000000000000e+00, 3.04000000000000000e+02, 8.51634157257224848e-03,
  5.22743532337373029e-01 ] # 28,3
                       , dtype=float)

                elif (seed == 4):
                    res_opt = np.array(
                          [ 2.80000000000000000e+01, 1.40000000000000000e+01, 2.01580003185462042e-01,
  1.52500000000000000e+01, 2.92750000000000000e+02, 8.39264463088176642e-03,
  5.27179984652617395e-01 ] # 28,4
                       , dtype=float)
        ################ END OPT + TEST ################
        elif is_OPT:
            print ("Optimised params")
            # Get better hyper-params
            if n_qubits == 20:
                if (seed == 0):
                    res_opt = np.array(
                          [2.000000000000000e+01,1.000000000000000e+01,1.379293313014652e-01,
                          1.400000000000000e+01,8.770000000000000e+02,6.772746596194328e-03,9.761443082361116e-01] # 20,0
                        , dtype=float)
                elif (seed == 1):
                    res_opt = np.array(
                          [2.0000000000000000e+01,1.0000000000000000e+01,2.8783274469043996e-01,
                          8.0000000000000000e+00,6.9000000000000000e+01,9.2330956886388078e-04,5.2642883932157569e-01] # 20,1
                        , dtype=float)

                elif (seed == 2):
                    res_opt = np.array(
                          [2.0000000000000000e+01,1.0000000000000000e+01,1.0536551430771036e-01,
                          1.0000000000000000e+01,9.8900000000000000e+02,4.3109846733713241e-04,8.7701792219291375e-01] # 20,2
                        , dtype=float)

                elif (seed == 3):
                    res_opt = np.array(
                          [2.0000000000000000e+01,1.0000000000000000e+01,1.0423545214103129e-01,
                          1.6000000000000000e+01,6.8000000000000000e+01,6.9110056065077412e-03,5.1287629290291625e-01] # 20,3
                        , dtype=float)

                elif (seed == 4):
                    res_opt = np.array(
                          [2.0000000000000000e+01,1.0000000000000000e+01,1.3164510173727237e-01,
                          1.9000000000000000e+01,9.7100000000000000e+02,2.1023477915894165e-03,7.7652259466571882e-01] # 20,4
                        , dtype=float)

                elif (seed == 5):
                    res_opt = np.array(
                          [2.000000000000000e+01,1.000000000000000e+01,1.983790413665049e-01,
                          7.000000000000000e+00,1.870000000000000e+02,8.345260413284723e-03,5.335363285208714e-01] # 20,5
                        , dtype=float)

                else:
                    print ("EERROR n_qubits is : " + str(n_qubits) + " with seed" + str(seed))
                    return 0

            elif n_qubits == 24:
                if (seed == 5):
                    res_opt = np.array(
                          [2.4000000000000000e+01,1.2000000000000000e+01,2.086990875809543e-01,
                          2.400000000000000e+01,9.330000000000000e+02,5.750352889953128e-03,6.201854402810651e-01] # 24,5
                        , dtype=float)

                elif (seed == 6):
                    res_opt = np.array(
                          [2.4000000000000000e+01,1.2000000000000000e+01,1.7548584518152849e-01,
                          1.0000000000000000e+01,3.9800000000000000e+02,7.2888822790476045e-03,9.7288726002346526e-01] # 24,6
                        , dtype=float)

                elif (seed == 7):
                    res_opt = np.array(
                          [2.4000000000000000e+01,1.2000000000000000e+01,1.8016632739969080e-01,
                          1.1000000000000000e+01,2.8600000000000000e+02,1.0634841086632695e-03,9.7592387484477428e-01] # 24,7
                        , dtype=float)

                elif (seed == 8):
                    res_opt = np.array(
                          [2.400000000000000e+01,1.200000000000000e+01,2.727222373426687e-01,
                          2.200000000000000e+01,4.530000000000000e+02,6.671559431293840e-03,8.298127092341986e-01] # 24,8
                        , dtype=float)

                elif (seed == 9):
                    res_opt = np.array(
                          [2.4000000000000000e+01,1.2000000000000000e+01,2.0906937632010314e-01,
                          2.4000000000000000e+01,6.5000000000000000e+02,3.9937657747336859e-03,5.2003186237253529e-01] # 24,9
                       , dtype=float)

                else:
                    print ("EERROR n_qubits is : " + str(n_qubits) + " with seed" + str(seed))
                    return 0

            elif n_qubits == 28:
                if (seed == 0):
                    res_opt = np.array(
                          [2.8000000000000000e+01,1.4000000000000000e+01,7.0017442061948805e-03,
                          7.0000000000000000e+00,3.1000000000000000e+02,6.7115397362500061e-03,6.3276343991833195e-01] # 28,0
                       , dtype=float)

                elif (seed == 1):
                    res_opt = np.array(
                          [2.8000000000000000e+01,1.4000000000000000e+01,1.9971283221559202e-01,
                          2.3000000000000000e+01,3.0900000000000000e+02,8.3995728796683409e-03,5.5405304641464770e-01] # 28,1
                       , dtype=float)

                elif (seed == 2):
                    res_opt = np.array(
                          [2.8000000000000000e+01,1.4000000000000000e+01,5.4676028526620297e-02,
                          1.2000000000000000e+01,9.9700000000000000e+02,1.2270564966696167e-03,5.6024259125284925e-01] # 28,2
                       , dtype=float)

                elif (seed == 3):
                    res_opt = np.array(
                          [2.800000000000000e+01,1.400000000000000e+01,4.233532475869142e-02,
                          6.000000000000000e+00,5.530000000000000e+02,8.405430923458371e-03,8.661621644069664e-01] # 28,3
                       , dtype=float)

                elif (seed == 4):
                    res_opt = np.array(
                          [2.8000000000000000e+01,1.4000000000000000e+01,2.3300486373514501e-01,
                          2.5000000000000000e+01,2.8000000000000000e+02,1.3007077509610715e-04,5.9852634070960953e-01] # 28,4
                       , dtype=float)
        ################ END OPT ONLY ################
        else:
            print ("Default params")
            res_opt = [num_qubits, num_qubits//2, 0.3, 10, 200, None, 0.8]

        # inputs
        prefix = str(num_qubits)+"20qubits_"

        if 0 <= seed <= 9:
            prefix +=f"0{seed}"
        else:
            prefix += str(seed)

        file_name = prefix + ".data"
        # Get Data
        result = process_file(folder_path, file_name)
        # Get Ham Qubit Size
        n_qubits=int(result[0])
        hamiltonian=result[1] # Need to load from Elena's files
        print_1(res_opt, hamiltonian, num_qubits, seed)

if __name__ == "__main__":
    main()
