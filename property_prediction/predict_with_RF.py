# Copyright Ryan-Rhys Griffiths and Aditya Raymond Thawani 2020
# Author: Ryan-Rhys Griffiths
"""
Property prediction on the photoswitch dataset using Random Forest.
"""

import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error

from data_utils import TaskDataLoader, transform_data, featurise_mols

PATH = '~/ml_physics/Photoswitches/dataset/photoswitches.csv'  # Change as appropriate
TASK = 'e_iso_pi'  # ['e_iso_pi', 'z_iso_pi', 'e_iso_n', 'z_iso_n']
representation = 'fragprints'  # ['fingerprints, 'fragments', 'fragprints']
use_pca = False
n_trials = 20  # number of random train/test splits to use
test_set_size = 0.2  # fraction of datapoints to use in the test set


if __name__ == '__main__':

    data_loader = TaskDataLoader(TASK, PATH)
    smiles_list, y = data_loader.load_property_data()

    X = featurise_mols(smiles_list, representation)

    if use_pca:
        n_components = 50
    else:
        n_components = None

    num_features = np.shape(X)[1]

    r2_list = []
    rmse_list = []
    mae_list = []

    print('\nBeginning training loop...')
    j = 0  # index for saving results

    for i in range(0, n_trials):

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_set_size, random_state=i)
        y_train = y_train.reshape(-1, 1)
        y_test = y_test.reshape(-1, 1)
        X_train, y_train, X_test, y_test, y_scaler = transform_data(X_train, y_train, X_test, y_test, n_components, use_pca)

        regr_rf = RandomForestRegressor(n_estimators=1000, max_depth=300, random_state=2)
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

        r2_list.append(score)
        rmse_list.append(rmse)
        mae_list.append(mae)

        np.savetxt(TASK + '/results/random_forest/_seed_'+str(j)+'_ypred_'+representation+'.txt', y_rf)
        np.savetxt(TASK + '/results/random_forest/_seed_'+str(j)+'_ytest.txt', y_test)

        j += 1

    r2_list = np.array(r2_list)
    rmse_list = np.array(rmse_list)
    mae_list = np.array(mae_list)
    print("\nmean R^2: {:.4f} +- {:.4f}".format(np.mean(r2_list), np.std(r2_list)/np.sqrt(len(r2_list))))
    print("mean RMSE: {:.4f} +- {:.4f}".format(np.mean(rmse_list), np.std(rmse_list)/np.sqrt(len(rmse_list))))
    print("mean MAE: {:.4f} +- {:.4f}\n".format(np.mean(mae_list), np.std(mae_list)/np.sqrt(len(mae_list))))
