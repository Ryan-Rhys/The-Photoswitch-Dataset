# Copyright Ryan-Rhys Griffiths and Aditya Raymond Thawani 2020
# Author: Ryan-Rhys Griffiths
"""
module for loading data
"""

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler


class DataLoader:
    """
    Data loader class
    """

    def __init__(self, task, path):
        """

        :param task: Property prediction task to load data from:
        ['thermal', 'e_iso_pi', 'z_iso_pi', 'e_iso_n', 'z_iso_n']
        :param path: Path to the photoswitches csv file
        """

        self.task = task
        self.path = path

    def load_property_data(self):
        """
        Load data corresponding to the property prediction task.
        :return: smiles_list, property_vals
        """

        df = pd.read_csv(self.path)
        smiles_list = df['SMILES'].to_list()

        if self.task == 'thermal':

            # Load the SMILES as x-values and the rate of thermal isomerisation as the y-values

            property_vals = df['rate of thermal isomerisation from Z-E in s-1'].to_numpy()

        elif self.task == 'e_iso_pi':

            #     Load the SMILES as x-values and the E isomer pi-pi* wavelength in nm as the y-values.
            #     108 molecules with valid experimental values as of 11 May 2020.

            property_vals = df['E isomer pi-pi* wavelength in nm'].to_numpy()

        elif self.task == 'z_iso_pi':

            #     Load the SMILES as x-values and the Z isomer pi-pi* wavelength in nm as the y-values.
            #     84 valid molecules for this property as of 11 May 2020.

            property_vals = df['Z isomer pi-pi* wavelength in nm'].to_numpy()

        elif self.task == 'e_iso_n':

            #     Load the SMILES as x-values and the E isomer n-pi* wavelength in nm as the y-values.
            #     96 valid molecules for this property as of 9 May 2020.

            property_vals = df['E isomer n-pi* wavelength in nm'].to_numpy()

        elif self.task == 'z_iso_n':

            #     Load the SMILES as x-values and the Z isomer n-pi* wavelength in nm as the y-values.
            #     93 valid molecules with this property as of 9 May 2020
            #     114 valid molecules with this property as of 16 May 2020

            property_vals = df['Z isomer n-pi* wavelength in nm'].to_numpy()

        else:
            raise Exception('Must specify a valid task')

        smiles_list = list(np.delete(np.array(smiles_list), np.argwhere(np.isnan(property_vals))))
        property_vals = np.delete(property_vals, np.argwhere(np.isnan(property_vals)))

        return smiles_list, property_vals


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
    y_train_scaled = y_scaler.fit_transform(y_train)
    y_test_scaled = y_scaler.transform(y_test)

    if use_pca:
        pca = PCA(n_components)
        X_train_scaled = pca.fit_transform(X_train_scaled)
        print('Fraction of variance retained is: ' + str(sum(pca.explained_variance_ratio_)))
        X_test_scaled = pca.transform(X_test_scaled)

    return X_train_scaled, y_train_scaled, X_test_scaled, y_test_scaled, y_scaler
