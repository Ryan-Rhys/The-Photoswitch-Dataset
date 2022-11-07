# Copyright Ryan-Rhys Griffiths 2020
# Author: Ryan-Rhys Griffiths
"""
module for loading data
"""

import numpy as np
import pandas as pd
from rdkit.Chem import AllChem, Descriptors, MolFromSmiles, MolToSmiles
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler


class TaskDataLoader:
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

    @staticmethod
    def load_dft_comparison_data(dft_path):
        """
        Load data for the purposes of comparison with DFT. Default task is the E isomer pi-pi* transition wavelength.
        Returns both the dft-computed values for the PBE0 level of theory and for the CAM-B3LYP levels of theory.
        Solvent_vals are the solvents used for the measurements.
        :param dft_path: path to the dft dataset dft_comparison.csv
        :return: smiles_list, solvent_vals, pbe0_vals, cam_vals, experimental_vals
        """

        df = pd.read_csv(dft_path)
        smiles_list = df['SMILES'].to_list()
        experimental_vals = df['Experiment'].to_numpy()
        pbe0_vals = df['PBE0'].to_numpy()
        cam_vals = df['CAM-B3LYP'].to_numpy()
        solvent_vals = df['Solvent'].to_list()

        return smiles_list, solvent_vals, pbe0_vals, cam_vals, experimental_vals

    @staticmethod
    def load_large_comparison_data(large_path):
        """
        Load electronic transition wavelengths for E isomer pi-pi* transitions from a large dataset of 8489 molecules.
        TD-DFT computations employed the LRC-wPBEh functional and 6–311 + G* data sets.
        :param large_path: str giving the path to the large dataset paper_allDB.csv from:
               https://www.nature.com/articles/s41597-019-0306-0
        :return: smiles_list, experiment_vals, dft_vals
        """

        df = pd.read_csv(large_path)
        smiles_list = df['SMILES'].to_list()
        experimental_vals = df['Experiment'].to_numpy()
        smiles_list = list(np.delete(np.array(smiles_list), np.argwhere(np.isnan(experimental_vals))))
        experimental_vals = np.delete(experimental_vals, np.argwhere(np.isnan(experimental_vals)))

        return smiles_list, experimental_vals


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
        X_train_scaled = pca.fit_transform(X_train)
        print('Fraction of variance retained is: ' + str(sum(pca.explained_variance_ratio_)))
        X_test_scaled = pca.transform(X_test)

    return X_train_scaled, y_train_scaled, X_test_scaled, y_test_scaled, y_scaler


def featurise_mols(smiles_list, representation, bond_radius=3, nBits=2048):
    """
    Featurise molecules according to representation
    :param smiles_list: list of molecule SMILES
    :param representation: str giving the representation. Can be 'fingerprints' or 'fragments'.
    :param bond_radius: int giving the bond radius for Morgan fingerprints. Default is 3
    :param nBits: int giving the bit vector length for Morgan fingerprints. Default is 2048
    :return: X, the featurised molecules
    """

    if representation == 'fingerprints':

        rdkit_mols = [MolFromSmiles(smiles) for smiles in smiles_list]
        X = [AllChem.GetMorganFingerprintAsBitVect(mol, bond_radius, nBits=nBits) for mol in rdkit_mols]
        X = np.asarray(X)

    elif representation == 'fragments':

        # descList[115:] contains fragment-based features only
        # (https://www.rdkit.org/docs/source/rdkit.Chem.Fragments.html)
        # NB latest version of RDKit has 7 more features that change indexing.

        fragments = {d[0]: d[1] for d in Descriptors.descList[115:]}
        X = np.zeros((len(smiles_list), len(fragments)))
        for i in range(len(smiles_list)):
            mol = MolFromSmiles(smiles_list[i])
            try:
                features = [fragments[d](mol) for d in fragments]
            except:
                raise Exception('molecule {}'.format(i) + ' is not canonicalised')
            X[i, :] = features

    else:

        # fragprints

        # convert to mol and back to smiles in order to make non-isomeric.

        rdkit_mols = [MolFromSmiles(smiles) for smiles in smiles_list]
        rdkit_smiles = [MolToSmiles(mol, isomericSmiles=False) for mol in rdkit_mols]
        rdkit_mols = [MolFromSmiles(smiles) for smiles in rdkit_smiles]
        X = [AllChem.GetMorganFingerprintAsBitVect(mol, 3, nBits=2048) for mol in rdkit_mols]
        X = np.asarray(X)

        fragments = {d[0]: d[1] for d in Descriptors.descList[115:]}
        X1 = np.zeros((len(smiles_list), len(fragments)))
        for i in range(len(smiles_list)):
            mol = MolFromSmiles(smiles_list[i])
            try:
                features = [fragments[d](mol) for d in fragments]
            except:
                raise Exception('molecule {}'.format(i) + ' is not canonicalised')
            X1[i, :] = features

        X = np.concatenate((X, X1), axis=1)

    return X
