
"""
This script trains a regression model (the default was XGBoost) to predict the performance
of quantum eigensolver implementation configurations using previously mined datasets of 
parameters (X) and classically computed energies (Y).

This file is part of the AccelerQ Project.
(2025) King's College London. CC BY 4.0.
- You must give appropriate credit, provide a link to the license, and indicate if changes 
  were made. You may do so in any reasonable manner, but not in any way that suggests 
  the licensor endorses you or your use.

"""

import sys
import os
import numpy as np
from kcl_util import (
    load_data_set,
    save_model,
    vec_to_fixed_size_vec,
    print_to_file,
    label_test_extra_split,
    update_train_test_predictions,
)

from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error

def train(folder, model_file, ham28_vec, regressor, cpu, params_ml, records_csv_file=None, params_q_size=13):
    # Make sure the precision is good enough  
    np.set_printoptions(precision=17)
    
    print ("Load data from folder " + folder)
    X, Y, X_extra, Y_extra = load_data_set(folder)

    csv_enabled = (
        isinstance(records_csv_file, str)
        and records_csv_file
        and os.path.exists(records_csv_file)
        and os.path.getsize(records_csv_file) > 0
    )
    if csv_enabled:
        label_test_extra_split(records_csv_file, folder)
    elif isinstance(records_csv_file, str) and records_csv_file:
        print(">> Skipping CSV updates: missing or empty " + records_csv_file)


    if len(X) == 0:
        print ("X is empty")
        raise ValueError("Size of X is 0")
    elif len(Y) == 0:
        print ("Y is empty")
        raise ValueError("Size of Y is 0")
    elif len(X_extra) == 0:
        print ("X extra is empty")
        raise ValueError("Size of X extra is 0")
    elif len(Y_extra) == 0:
        print ("Y extra is empty")
        raise ValueError("Size of Y extra is 0")
    elif len(X) != len(Y):
        print ("X and Y are of a differnt len:" + len(X) + " != " + len(Y) )
        raise ValueError("X and Y are of a different len")
    elif len(X_extra) != len(Y_extra):
        print ("X extra and Y extra are of a differnt len:" + len(X_extra) + " != " + len(Y_extra) )
        raise ValueError("X extra and Y extra are of a different len")
    else:
        print (">> Data loaded okay")

    # max vector width = ham features + QE hyperparams (params_q_size)
    max_size = max(len(ham28_vec), max([len(vector) for vector in X])) + params_q_size
    print (">> Size of each vector is: " + str(max_size) + " with ham28 vec size " + str(len(ham28_vec)))

    for i in range(len(X)):
        size_before = len(X[i])
        X[i] = vec_to_fixed_size_vec(max_size, X[i])
        size_after = len(X[i])
        if size_after != max_size:
            print("Size before and after: " + size_before + "," + size_after + " with max size:" + str(max_size))
            raise ValueError("Size of X - padding failed. Itr: " + str(i))

    for i in range(len(X_extra)):
        size_before = len(X_extra[i])
        X_extra[i] = vec_to_fixed_size_vec(max_size, X_extra[i])
        size_after = len(X_extra[i])
        if size_after != max_size:
            print("Size before and after: " + size_before + "," + size_after + " with max size:" + str(max_size))
            raise ValueError("Size of X_extra - padding failed. Itr: " + str(i))
    
    # Split dataset into train and test sets - 
    # with recoverable positions for records used in testing
    indices = list(range(len(X)))
    x_train, x_test, y_train, y_test, idx_train, idx_test = train_test_split(
        X, Y, indices, test_size=0.20, random_state=42
        )

    chunk_size=9
    num_chunks=len(X)//chunk_size
    print (">>> Train the model with size ", str(chunk_size))

    model = regressor(**params_ml, random_state=42)
    if cpu == 1:
        for chunk in range(num_chunks):
            start_idx = chunk * chunk_size
            end_idx = (chunk + 1) * chunk_size
            x_chunk = X[start_idx:end_idx]
            y_chunk = Y[start_idx:end_idx]

            print_to_file(">>>> Start training chunk " + str(chunk))
            if chunk == 0:
                model.fit(x_chunk, y_chunk)
            else:
                model.n_estimators += params_ml['n_estimators']
                model.fit(x_chunk, y_chunk, xgb_model=model)
    else:
        model.fit(x_train,y_train)

    # Evaluate on test data
    y_pred_test = model.predict(x_test)
    mse = mean_squared_error(y_test, y_pred_test)
    print (">>>>>>>>>>>>> Mean Squared Error:", mse, ":: Predictions:", y_pred_test, ":: True Values:", y_test)
    abe = mean_absolute_error(y_test, y_pred_test)
    print (">>>>>>>>>>>>> Mean ABS Error:", abe)

    # Evaluate on extra holdout data
    y_pred_extra = model.predict(X_extra)
    mse = mean_squared_error(Y_extra, y_pred_extra)
    print (">>>>>>>>>>>>> Mean Squared Error:", mse, ":: Predictions:", y_pred_extra, ":: True Values:", Y_extra)
    abe = mean_absolute_error(Y_extra, y_pred_extra)
    print (">>>>>>>>>>>>> Mean ABS Error:", abe)

    if csv_enabled:
        update_train_test_predictions(
            records_csv_file, folder, idx_train, idx_test,
            y_pred_test, y_test, y_pred_extra, Y_extra,
        )

    save_model(model_file, model)

    print (">>> Stat. >>> Size of TRAIN set: " + str(len(x_train)))
    print (">>> Stat. >>> Size of TEST set: " + str(len(x_test)))
    print (">>> Stat. >>> Size of EXTRA TEST set: " + str(len(X_extra)))

    return max_size
