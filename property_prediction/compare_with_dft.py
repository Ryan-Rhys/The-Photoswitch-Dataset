# Copyright Ryan-Rhys Griffiths and Aditya Raymond Thawani 2020
# Author: Ryan-Rhys Griffiths
"""
Property prediction comparison with DFT
"""

import numpy as np
from rdkit.Chem import AllChem, MolFromSmiles
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error

from data_utils import transform_data, dft_train_test_split

PATH = '~/ml_physics/Photoswitches/dataset/photoswitches.csv'  # Change as appropriate
TASK = 'z_iso_n'  # ['e_iso_pi', 'z_iso_pi', 'e_iso_n', 'z_iso_n']
use_pca = False


if __name__ == '__main__':

    if TASK == 'e_iso_pi':
        X_train, X_test, y_train, y_test, dft_vals = dft_train_test_split(PATH, TASK)
    elif TASK == 'z_iso_pi':
        X_train, X_test, y_train, y_test, dft_vals = dft_train_test_split(PATH, TASK)
    elif TASK == 'e_iso_n':
        X_train, X_test, y_train, y_test, dft_vals = dft_train_test_split(PATH, TASK)
    elif TASK == 'z_iso_n':
        X_train, X_test, y_train, y_test, dft_vals = dft_train_test_split(PATH, TASK)
    else:
        raise Exception('Must specify a valid task')

    rdkit_train_mols = [MolFromSmiles(smiles) for smiles in X_train]
    X_train = [AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=512) for mol in rdkit_train_mols]
    X_train = np.asarray(X_train)

    rdkit_test_mols = [MolFromSmiles(smiles) for smiles in X_test]
    X_test = [AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=512) for mol in rdkit_test_mols]
    X_test = np.asarray(X_test)

    X_train, y_train, X_test, y_test, y_scaler = transform_data(X_train, y_train, X_test, y_test)

    regr_rf = RandomForestRegressor(n_estimators=100, max_depth=30, random_state=2)
    regr_rf.fit(X_train, y_train)

    # Predict on new data
    y_rf = regr_rf.predict(X_test)
    y_rf = y_scaler.inverse_transform(y_rf)
    y_test = y_scaler.inverse_transform(y_test)
    score = r2_score(y_test, y_rf)
    rmse = np.sqrt(mean_squared_error(y_test, y_rf))
    mae = mean_absolute_error(y_test, y_rf)

    dft_rmse = np.sqrt(mean_squared_error(y_test, dft_vals))
    dft_mae = mean_absolute_error(y_test, dft_vals)
    dft_score = r2_score(y_test, dft_vals)

    print("\nDFT RMSE: {:.3f}".format(dft_rmse))
    print("DFT MAE: {:.3f}".format(dft_mae))
    print("R^2: {:.3f}".format(dft_score))

    print("\nRMSE: {:.3f}".format(rmse))
    print("MAE: {:.3f}".format(mae))
    print("R^2: {:.3f}".format(score))

