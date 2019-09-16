import numpy as np
import pandas as pd

def data_processing():
    # low_memory: Internally process the file in chunks, resulting in lower memory use while parsing, but possibly mixed type inference.
    # To ensure no mixed types either set False, or specify the type with the dtype parameter.

    # na_values: Additional strings to recognize as NA/NaN.
    data = pd.read_csv('heart_disease.csv', low_memory=False, sep=',', na_values='?').values

    # number of data
    N = data.shape[0]

    # Modify a sequence in-place by shuffling its contents.
    # This function only shuffles the array along the first axis of a multi-dimensional array.
    np.random.shuffle(data)

    # separate data into train set, validation set, test set
    ntr = int(np.round(N * 0.8))
    nval = int(np.round(N * 0.15))
    ntest = N - ntr - nval

    # modify training, validation, and test set to matrix without label
    # modify the label of training, validation, and test set
    x_train = np.append([np.ones(ntr)], data[:ntr].T[:-1], axis=0).T
    y_train = data[:ntr].T[-1].T
    x_val = np.append([np.ones(nval)], data[ntr:ntr + nval].T[:-1], axis=0).T
    y_val = data[ntr:ntr + nval].T[-1].T
    x_test = np.append([np.ones(ntest)], data[-ntest:].T[:-1], axis=0).T
    y_test = data[-ntest:].T[-1].T
    return x_train, y_train, x_val, y_val, x_test, y_test


