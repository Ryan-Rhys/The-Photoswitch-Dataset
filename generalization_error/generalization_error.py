# Author: Ryan-Rhys Griffiths
"""
Script to test generalization performance of a model trained on the E isomer pi-pi* transition wavelengths of a large
dataset of 6,142 molecules from Beard et al. 2019 https://www.nature.com/articles/s41597-019-0306-0
Generalisation performance is gauged relative to the full photoswitch dataset. We also test generalization performance
when this dataset is leveraged as additional training data.
"""

import argparse

import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from sklearn.model_selection import train_test_split

from data_utils import TaskDataLoader, transform_data, featurise_mols


def main(path, path_to_large_dataset, task, representation, test_set_size, augment_photo_dataset, n_trials):
    """
    :param path: str giving path to the photoswitches.csv file.
    :param path_to_large_dataset: str giving path to paper_allDB.csv file
    :param task: str specifying the task. Always e_iso_pi for the generalization experiment
    :param representation: str specifying the molecular representation. One of [fingerprints, fragments, fragprints].'
    :param test_set_size: float in range [0, 1] specifying fraction of dataset to use as test set
    :param augment_photo_dataset: If True augment the photoswitch dataset with the Beard et al. 2019 dataset
    :param n_trials: int specifying the number of random train/test splits.
    """

    data_loader = TaskDataLoader(task, path)

    photo_smiles_list, y_vals_photo = data_loader.load_property_data()
    beard_smiles_list, y_vals_beard = data_loader.load_large_comparison_data(path_to_large_dataset)

    r2_list = []
    rmse_list = []
    mae_list = []

    if not augment_photo_dataset:
        # test set is now fixed
        n_trials = 1
        # We train on the Beard dataset and test on the photoswitch dataset
        X_train = featurise_mols(beard_smiles_list, representation)
        X_test = featurise_mols(photo_smiles_list, representation)
        y_train = y_vals_beard
        y_test = y_vals_photo

    for i in range(0, n_trials):

        if augment_photo_dataset:
            # We add the Beard dataset as additional training data
            X_train, X_test, y_train, y_test = train_test_split(photo_smiles_list, y_vals_photo, test_size=test_set_size, random_state=i)
            X_train = X_train + beard_smiles_list
            y_train = np.concatenate((y_train, y_vals_beard))
            X_train = featurise_mols(X_train, representation)
            X_test = featurise_mols(X_test, representation)

        y_train = y_train.reshape(-1, 1)
        y_test = y_test.reshape(-1, 1)
        X_train, y_train, X_test, y_test, y_scaler = transform_data(X_train, y_train, X_test, y_test)

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

        r2_list.append(score)
        rmse_list.append(rmse)
        mae_list.append(mae)

        print("\nR^2: {:.3f}".format(score))
        print("RMSE: {:.3f}".format(rmse))
        print("MAE: {:.3f}".format(mae))

    r2_list = np.array(r2_list)
    rmse_list = np.array(rmse_list)
    mae_list = np.array(mae_list)

    print("\nmean R^2: {:.4f} +- {:.4f}".format(np.mean(r2_list), np.std(r2_list) / np.sqrt(len(r2_list))))
    print("mean RMSE: {:.4f} +- {:.4f}".format(np.mean(rmse_list), np.std(rmse_list) / np.sqrt(len(rmse_list))))
    print("mean MAE: {:.4f} +- {:.4f}\n".format(np.mean(mae_list), np.std(mae_list) / np.sqrt(len(mae_list))))


if __name__ == '__main__':

    parser = argparse.ArgumentParser()

    parser.add_argument('-p', '--path', type=str, default='../dataset/photoswitches.csv',
                        help='Path to the photoswitches.csv file.')
    parser.add_argument('-pl', '--path_to_large_dataset', type=str, default='../dataset/paper_allDB.csv',
                        help='str giving path to paper_allDB.csv file')
    parser.add_argument('-t', '--task', type=str, default='e_iso_pi',
                        help='str specifying the task. Always e_iso_pi in the case of the '
                             'generalization error experiment')
    parser.add_argument('-r', '--representation', type=str, default='fragprints',
                        help='str specifying the molecular representation. '
                             'One of [fingerprints, fragments, fragprints].')
    parser.add_argument('-ts', '--test_set_size', type=float, default=0.2,
                        help='float in range [0, 1] specifying fraction of dataset to use as test set')
    parser.add_argument('-au', '--augment_photo_dataset', type=bool, default=False,
                        help='If True augment the photoswitch dataset with the Beard et al. 2019 dataset.')
    parser.add_argument('-n', '--n_trials', type=int, default=20,
                        help='int specifying number of random train/test splits to use')

    args = parser.parse_args()

    main(args.path, args.path_to_large_dataset, args.task, args.representation, args.test_set_size,
         args.augment_photo_dataset, args.n_trials,)
