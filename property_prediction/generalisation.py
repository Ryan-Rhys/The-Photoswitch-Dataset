# Author: Ryan-Rhys Griffiths
"""
Script to test generalisation performance of a model trained on the E isomer pi-pi* transition wavelengths of a large
dataset of 6,142 molecules. Generalisation performance is gauged relative to the full photoswitch dataset.
"""

import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error

from data_utils import TaskDataLoader, transform_data, featurise_mols

PATH = '~/ml_physics/Photoswitches/dataset/photoswitches.csv'  # Change as appropriate
LARGE_PATH = '~/ml_physics/Photoswitches/dataset/paper_allDB.csv'
TASK = 'e_iso_pi'  # ['e_iso_pi', 'z_iso_pi', 'e_iso_n', 'z_iso_n']
representation = 'fragprints'  # ['fingerprints, 'fragments', 'fragprints']
test_set_size = 0.  # fraction of datapoints to use in the test set


if __name__ == '__main__':

    data_loader = TaskDataLoader(TASK, PATH)

    test_smiles_list, y_test = data_loader.load_property_data()
    large_smiles_list, y_train = data_loader.load_large_comparison_data(LARGE_PATH)

    X_train = featurise_mols(large_smiles_list, representation)
    X_test = featurise_mols(test_smiles_list, representation)

    y_train = y_train.reshape(-1, 1)
    y_test = y_test.reshape(-1, 1)
    X_train, y_train, X_test, y_test, y_scaler = transform_data(X_train, y_train, X_test, y_test)

    regr_rf = RandomForestRegressor(n_estimators=200, max_depth=30, random_state=2)
    regr_rf.fit(X_train, y_train)

    # Output Standardised RMSE and RMSE on Train Set

    y_pred_train = regr_rf.predict(X_train)
    train_rmse_stan = np.sqrt(mean_squared_error(y_train, y_pred_train))
    train_rmse = np.sqrt(mean_squared_error(y_scaler.inverse_transform(y_train), y_scaler.inverse_transform(y_pred_train)))
    print("\nStandardised Train RMSE: {:.3f}".format(train_rmse_stan))
    print("Train RMSE: {:.3f}".format(train_rmse))

    # Predict on new data
    y_rf = regr_rf.predict(X_test)
    y_rf = y_scaler.inverse_transform(y_rf)
    y_test = y_scaler.inverse_transform(y_test)
    score = r2_score(y_test, y_rf)
    rmse = np.sqrt(mean_squared_error(y_test, y_rf))
    mae = mean_absolute_error(y_test, y_rf)

    print("\nR^2: {:.3f}".format(score))
    print("RMSE: {:.3f}".format(rmse))
    print("MAE: {:.3f}".format(mae))
