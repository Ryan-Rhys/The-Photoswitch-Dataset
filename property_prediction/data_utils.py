# Copyright Ryan-Rhys Griffiths and Aditya Raymond Thawani 2020
# Author: Ryan-Rhys Griffiths
"""
module for loading data
"""

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler


def load_thermal_data(path):
    """
    Load the SMILES as x-values and the rate of thermal isomerisation as the y-values.

    :param path: path to dataset
    :return: SMILES, rate of thermal isomerisation values
    """

    df = pd.read_csv(path)
    smiles_list = df['SMILES'].to_list()
    thermal_vals = df['rate of thermal isomerisation from Z-E in s-1'].to_numpy()

    # Remove molecules for which the value of thermal isomerisation rate is infinity or not available

    smiles_list = smiles_list[:65]
    thermal_vals = thermal_vals[:65]

    return smiles_list, thermal_vals


def load_e_iso_pi_data(path):
    """
    Load the SMILES as x-values and the E isomer pi-pi* wavelength in nm as the y-values.
    108 molecules with valid experimental values as of 11 May 2020. Index 42 is a nan value.

    :param path: path to dataset
    :return: SMILES, property
    """

    df = pd.read_csv(path)
    smiles_list = df['SMILES'].to_list()
    smiles_list.remove('CC(NC(C=C1)=CC(OC)=C1/N=N/C2=C(OC)C=C(NC(C)=O)C=C2)=O')
    e_iso_pi_vals = df['E isomer pi-pi* wavelength in nm'].to_numpy()
    e_iso_pi_vals = np.delete(e_iso_pi_vals, 42)

    return smiles_list, e_iso_pi_vals


def load_e_iso_n_data(path):
    """
    Load the SMILES as x-values and the E isomer n-pi* wavelength in nm as the y-values.
    96 valid molecules for this property as of 9 May 2020.

    :param path: path to dataset
    :return: SMILES, property
    """

    df = pd.read_csv(path)
    smiles_list = df['SMILES'].to_list()
    smiles_list = smiles_list[0:14] + smiles_list[26:49] + smiles_list[50:]
    e_iso_n_vals = df['E isomer n-pi* wavelength in nm'].to_numpy()
    selection = np.concatenate((np.arange(14, 26, 1), np.array([49])))
    e_iso_n_vals = np.delete(e_iso_n_vals, selection)

    return smiles_list, e_iso_n_vals


def load_z_iso_pi_data(path):
    """
    Load the SMILES as x-values and the Z isomer pi-pi* wavelength in nm as the y-values.
    84 valid molecules for this property as of 11 May 2020.

    :param path: path to dataset
    :return: SMILES, property
    """

    df = pd.read_csv(path)
    smiles_list = df['SMILES'].to_list()

    # Remove NaN indices
    smiles_list = smiles_list[0:12] + smiles_list[26:36] + [smiles_list[37]] + smiles_list[39:42] + [smiles_list[44]] + [smiles_list[48]] + smiles_list[53:]
    z_iso_pi_vals = df['Z isomer pi-pi* wavelength in nm'].to_numpy()
    selection = np.concatenate((np.arange(45, 48, 1), np.arange(49, 53, 1)))
    selection = np.concatenate((np.arange(42, 44, 1), selection))
    selection = np.concatenate((np.array([38]), selection))
    selection = np.concatenate((np.array([36]), selection))
    selection = np.concatenate((np.arange(12, 26, 1), selection))
    z_iso_pi_vals = np.delete(z_iso_pi_vals, selection)

    return smiles_list, z_iso_pi_vals


def load_z_iso_n_data(path):
    """
    Load the SMILES as x-values and the Z isomer n-pi* wavelength in nm as the y-values.
    93 valid molecules with this property as of 9 May 2020

    :param path: path to dataset
    :return: SMILES, property
    """

    df = pd.read_csv(path)
    smiles_list = df['SMILES'].to_list()
    smiles_list = smiles_list[0:12] + smiles_list[15:43] + smiles_list[44:52] + smiles_list[53:]
    z_iso_n_vals = df['Z isomer n-pi* wavelength in nm'].to_numpy()
    z_iso_n_vals = np.delete(z_iso_n_vals, np.array([12, 13, 14, 43, 52]))

    return smiles_list, z_iso_n_vals


def dft_train_test_split(path, task):
    """
    Load the train/test split data required for comparison against DFT.
    :param path: path to dataset
    :param task: string specifiying the property prediction task ['e_iso_pi', 'z_iso_pi', 'e_iso_n', 'z_iso_n']
    :return: X_train, X_test, y_train, y_test, dft_vals
    """

    df = pd.read_csv(path)
    smiles_list = df['SMILES'].to_list()

    if task == 'e_iso_pi':

        exp_vals = df['E isomer pi-pi* wavelength in nm'].to_numpy()
        dft_vals = df['Closest DFT E isomer pi-pi* wavelength in nm'].to_numpy()

    elif task == 'z_iso_pi':

        exp_vals = df['Z isomer pi-pi* wavelength in nm'].to_numpy()
        dft_vals = df['Closest DFT Z isomer pi-pi* wavelength in nm'].to_numpy()

    elif task == 'e_iso_n':

        exp_vals = df['E isomer n-pi* wavelength in nm'].to_numpy()
        dft_vals = df['Closest DFT E isomer n-pi* wavelength in nm'].to_numpy()

    elif task == 'z_iso_n':

        exp_vals = df['Z isomer n-pi* wavelength in nm'].to_numpy()
        dft_vals = df['Closest DFT Z isomer n-pi* wavelength in nm'].to_numpy()

    else:
        raise Exception('Must provide a valid task')

    y_train_with_missing_vals = np.delete(exp_vals, np.argwhere(~np.isnan(dft_vals)))
    y_train = np.delete(y_train_with_missing_vals, np.argwhere(np.isnan(y_train_with_missing_vals)))

    y_test = np.delete(exp_vals, np.concatenate((np.argwhere(np.isnan(dft_vals)), np.argwhere(np.isnan(exp_vals)))))
    X_test = list(np.delete(np.array(smiles_list),
                            np.concatenate((np.argwhere(np.isnan(dft_vals)), np.argwhere(np.isnan(exp_vals))))))
    X_train_with_missing_vals = np.delete(np.array(smiles_list), np.argwhere(~np.isnan(dft_vals)))
    X_train = list(np.delete(X_train_with_missing_vals, np.argwhere(np.isnan(y_train_with_missing_vals))))
    dft_vals = np.delete(dft_vals,
                         np.concatenate((np.argwhere(np.isnan(dft_vals)), np.argwhere(np.isnan(exp_vals)))))

    return X_train, X_test, y_train, y_test, dft_vals


def transform_data(X_train, y_train, X_test, y_test, n_components=None, use_pca=False):
    """
    Apply feature scaling, dimensionality reduction to the data. Return the standardised and low-dimensional train and
    test sets together with the scaler object for the target values.

    :param X_train: input train data
    :param y_train: train labels
    :param X_test: input test data
    :param y_test: test labels
    :param n_components: number of principal components to keep when use_pca = True
    :param use_pca: Whether or not to use PCA
    :return: X_train_scaled, y_train_scaled, X_test_scaled, y_test_scaled, y_scaler
    """

    x_scaler = StandardScaler()
    X_train_scaled = x_scaler.fit_transform(X_train)
    X_test_scaled = x_scaler.transform(X_test)
    y_scaler = StandardScaler()
    y_train_scaled = y_scaler.fit_transform(y_train.reshape(-1, 1))
    y_test_scaled = y_scaler.transform(y_test.reshape(-1, 1))

    if use_pca:
        pca = PCA(n_components)
        X_train_scaled = pca.fit_transform(X_train_scaled)
        print('Fraction of variance retained is: ' + str(sum(pca.explained_variance_ratio_)))
        X_test_scaled = pca.transform(X_test_scaled)

    return X_train_scaled, y_train_scaled, X_test_scaled, y_test_scaled, y_scaler
