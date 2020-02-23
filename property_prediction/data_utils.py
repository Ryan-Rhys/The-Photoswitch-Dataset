"""
Script for loading data
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

    # Remove molecules for which the value of thermal isomerisation rate is infinity

    smiles_list.remove('C1(/N=N\\2)=CC=CC=C1CCCC3=C2C=CC=C3')  # the infinity value
    smiles_list.remove('CN1C(/N=N/C2=CC=CC=C2)=CC=C1')
    smiles_list.remove('CN(N=C1)C=C1/N=N/C2=C(F)C=CC=C2F')
    smiles_list.remove('CN(N=C1C)C(C)=C1/N=N/C2=C(Cl)C=CC=C2Cl')
    smiles_list.remove('CN(N=C1C)C(C)=C1/N=N/C2=C(OC)C=CC=C2OC')
    smiles_list.remove('FC1=CC=CC=C1/N=N/C2=C(F)C=CC=C2')
    smiles_list.remove('CC1=C(/N=N/C2=CC=CC=C2)C=NC=C1')
    thermal_vals = np.delete(thermal_vals, 51)  # the infinity value
    thermal_vals = np.delete(thermal_vals, 34)
    thermal_vals = np.delete(thermal_vals, 18)
    thermal_vals = np.delete(thermal_vals, 16)
    thermal_vals = np.delete(thermal_vals, 15)
    thermal_vals = np.delete(thermal_vals, 14)
    thermal_vals = np.delete(thermal_vals, 10)
    thermal_vals = thermal_vals.astype(np.float)

    return smiles_list, thermal_vals


def load_e_iso_pi_data(path):
    """
    Load the SMILES as x-values and the E isomer pi-pi* wavelength in nm as the y-values.

    :param path: path to dataset
    :return: SMILES, property
    """

    df = pd.read_csv(path)
    smiles_list = df['SMILES'].to_list()
    smiles_list = smiles_list[:42]
    e_iso_pi_vals = df['E isomer pi-pi* wavelength in nm'].to_numpy()[0:42]
    smiles_list.remove('C12=CC=CC=C1CCC3=CC=CC=C3/N=N\\2')
    smiles_list.remove('CN(N=C1)C=C1/N=N/C2=C(F)C=CC=C2F')
    e_iso_pi_vals = np.delete(e_iso_pi_vals, 31)
    e_iso_pi_vals = np.delete(e_iso_pi_vals, 15)

    return smiles_list, e_iso_pi_vals


def transform_data(X_train, y_train, X_test, y_test, n_components):
    """
    Apply feature scaling, dimensionality reduction to the data. Return the standardised and low-dimensional train and
    test sets together with the scaler object for the target values.

    :param X_train: input train data
    :param y_train: train labels
    :param X_test: input test data
    :param y_test: test labels
    :param n_components: number of principal components to keep
    :return: X_train_scaled, y_train_scaled, X_test_scaled, y_test_scaled, y_scaler
    """

    x_scaler = StandardScaler()
    X_train_scaled = x_scaler.fit_transform(X_train)
    X_test_scaled = x_scaler.transform(X_test)
    y_scaler = StandardScaler()
    y_train_scaled = y_scaler.fit_transform(y_train.reshape(-1, 1))
    y_test_scaled = y_scaler.transform(y_test.reshape(-1, 1))
    pca = PCA(n_components)
    X_train_scaled = pca.fit_transform(X_train_scaled)
    print('Fraction of variance retained is: ' + str(sum(pca.explained_variance_ratio_)))
    X_test_scaled = pca.transform(X_test_scaled)
    return X_train_scaled, y_train_scaled, X_test_scaled, y_test_scaled, y_scaler
