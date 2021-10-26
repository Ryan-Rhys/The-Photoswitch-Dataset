# Copyright Ryan-Rhys Griffiths and Aditya Raymond Thawani 2020
# Author: Ryan-Rhys Griffiths
"""
Script for hyperparameter tuning.
"""

import argparse

from hpsklearn import HyperoptEstimator, random_forest_regression
from sklearn.model_selection import train_test_split

from data_utils import TaskDataLoader, transform_data, featurise_mols


def main(path, task, representation, use_pca):
    """
    :param path: str specifying path to dataset.
    :param task: str specifying the task. One of ['e_iso_pi', 'z_iso_pi', 'e_iso_n', 'z_iso_n']
    :param representation: str specifying the molecular representation. One of ['fingerprints, 'fragments', 'fragprints']
    :param use_pca: bool. If True apply PCA to perform Principal Components Regression.
    """

    data_loader = TaskDataLoader(task, path)
    smiles_list, y = data_loader.load_property_data()

    X = featurise_mols(smiles_list, representation)

    if use_pca:
        n_components = 50
    else:
        n_components = None

    # Set random state to be different to the splits used for evaluation

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=30)

    y_train = y_train.reshape(-1, 1)
    y_test = y_test.reshape(-1, 1)

    X_train, y_train, _, _, y_scaler = transform_data(X_train, y_train, X_test, y_test, n_components, use_pca)

    estim = HyperoptEstimator(regressor=random_forest_regression('my_RF'), max_evals=1000)
    estim.fit(X_train, y_train, valid_size=0.1, n_folds=5, cv_shuffle=True)
    print(estim.best_model())
    with open(f'saved_hypers/RF/tuning_for_{task}', 'w') as f:
        print(estim.best_model(), file=f)


if __name__ == '__main__':

    parser = argparse.ArgumentParser()

    parser.add_argument('-p', '--path', type=str, default='../dataset/photoswitches.csv',
                        help='Path to the photoswitches.csv file.')
    parser.add_argument('-t', '--task', type=str, default='e_iso_n',
                        help='str specifying the task. One of [e_iso_pi, z_iso_pi, e_iso_n, z_iso_n].')
    parser.add_argument('-r', '--representation', type=str, default='fragprints',
                        help='str specifying the molecular representation. '
                             'One of [fingerprints, fragments, fragprints].')
    parser.add_argument('-pca', '--use_pca', type=bool, default=False,
                        help='If True apply PCA to perform Principal Components Regression.')

    args = parser.parse_args()

    main(args.path, args.task, args.representation, args.use_pca)
