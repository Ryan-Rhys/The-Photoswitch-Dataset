# Copyright Ryan-Rhys Griffiths 2020
# Author: Ryan-Rhys Griffiths
"""
Script for training a model to predict properties in the photoswitch dataset using multioutput
Gaussian Process Regression with the intrinsic model of coregionalisation (ICM) (Bonilla and Williams, 2008).
"""

import argparse

import gpflow
from gpflow.ci_utils import ci_niter
from gpflow.mean_functions import Constant
from gpflow.utilities import print_summary, set_trainable
from matplotlib import pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error

from data_utils import TaskDataLoader, featurise_mols
from kernels import Tanimoto


def main(path, task, representation, use_pca, n_trials, test_set_size):
    """
    Train a multioutput GP simultaneously on all tasks of the photoswitch dataset.

    :param path: str specifying path to dataset.
    :param task: str specifying the task. One of ['e_iso_pi', 'z_iso_pi', 'e_iso_n', 'z_iso_n']
    :param representation: str specifying the molecular representation. One of ['fingerprints, 'fragments', 'fragprints']
    :param use_pca: bool. If True apply PCA to perform Principal Components Regression.
    :param n_trials: int specifying number of random train/test splits to use
    :param test_set_size: float in range [0, 1] specifying fraction of dataset to use as test set
    """

    # If True we perform Principal Components Regression

    if use_pca:
        n_components = 100
    else:
        n_components = None

    data_loader_e_iso_pi = TaskDataLoader('e_iso_pi', path)
    data_loader_z_iso_pi = TaskDataLoader('z_iso_pi', path)
    data_loader_e_iso_n = TaskDataLoader('e_iso_n', path)
    data_loader_z_iso_n = TaskDataLoader('z_iso_n', path)

    smiles_list_e_iso_pi, y_e_iso_pi = data_loader_e_iso_pi.load_property_data()
    smiles_list_z_iso_pi, y_z_iso_pi = data_loader_z_iso_pi.load_property_data()
    smiles_list_e_iso_n, y_e_iso_n = data_loader_e_iso_n.load_property_data()
    smiles_list_z_iso_n, y_z_iso_n = data_loader_z_iso_n.load_property_data()

    y_e_iso_pi = y_e_iso_pi.reshape(-1, 1)
    y_z_iso_pi = y_z_iso_pi.reshape(-1, 1)
    y_e_iso_n = y_e_iso_n.reshape(-1, 1)
    y_z_iso_n = y_z_iso_n.reshape(-1, 1)

    X_e_iso_pi = featurise_mols(smiles_list_e_iso_pi, representation)
    X_z_iso_pi = featurise_mols(smiles_list_z_iso_pi, representation)
    X_e_iso_n = featurise_mols(smiles_list_e_iso_n, representation)
    X_z_iso_n = featurise_mols(smiles_list_z_iso_n, representation)

    output_dim = 4  # Number of outputs
    rank = 1  # Rank of W
    feature_dim = len(X_e_iso_pi[0, :])

    tanimoto_active_dims = [i for i in range(feature_dim)]  # active dims for Tanimoto base kernel.

    r2_list = []
    rmse_list = []
    mae_list = []

    print('\nBeginning training loop...')

    for i in range(0, n_trials):

        if task == 'e_iso_pi':
            X_task = X_e_iso_pi
            y_task = y_e_iso_pi
        elif task == 'z_iso_pi':
            X_task = X_z_iso_pi
            y_task = y_z_iso_pi
        elif task == 'e_iso_n':
            X_task = X_e_iso_n
            y_task = y_e_iso_n
        else:
            X_task = X_z_iso_n
            y_task = y_z_iso_n

        X_train, X_test, y_train, y_test = train_test_split(X_task, y_task, test_size=test_set_size, random_state=i)

        X_train = X_train.astype(np.float64)
        X_test = X_test.astype(np.float64)

        def test_leakage(x1, x2, x3, y1, y2, y3, X_test):
            """
            Function to prevent test leakage in train/test splits for multitask learning
            param: x1, x2, x3: input molecules for other tasks
            param: y1, y2, y3: labels for other tasks
            param: X_test: the test molecules
            """

            other_tasks = [x1, x2, x3]
            other_labels = [y1, y2, y3]
            for i in range(len(other_tasks)):
                indices_to_delete = []
                for j in range(len(other_tasks[i])):
                    other_mol = other_tasks[i][j]
                    if np.any([np.array_equal(other_mol, mol) for mol in X_test]) == True:
                        indices_to_delete.append(j)
                indices_to_delete.reverse()
                for index in indices_to_delete:
                    other_tasks[i] = np.delete(other_tasks[i], index, axis=0)
                    other_labels[i] = np.delete(other_labels[i], index, axis=0)

            x1, x2, x3 = other_tasks[0], other_tasks[1], other_tasks[2]
            y1, y2, y3 = other_labels[0], other_labels[1], other_labels[2]

            return x1, x2, x3, y1, y2, y3

        if task == 'e_iso_pi':
            X_z_iso_pi, X_e_iso_n, X_z_iso_n, y_z_iso_pi, y_e_iso_n, y_z_iso_n = \
                test_leakage(X_z_iso_pi, X_e_iso_n, X_z_iso_n, y_z_iso_pi, y_e_iso_n, y_z_iso_n, X_test)

            # Augment the input with zeroes, ones, twos, threes to indicate the required output dimension
            X_augmented = np.vstack((np.append(X_train, np.zeros((len(X_train), 1)), axis=1),
                                     np.append(X_z_iso_pi, np.ones((len(X_z_iso_pi), 1)), axis=1),
                                     np.append(X_e_iso_n, np.ones((len(X_e_iso_n), 1)) * 2, axis=1),
                                     np.append(X_z_iso_n, np.ones((len(X_z_iso_n), 1)) * 3, axis=1)))

            X_test = np.append(X_test, np.zeros((len(X_test), 1)), axis=1)
            X_train = np.append(X_train, np.zeros((len(X_train), 1)), axis=1)

            # Augment the Y data with zeroes, ones, twos and threes that specify a likelihood from the list of likelihoods
            Y_augmented = np.vstack((np.hstack((y_train, np.zeros_like(y_train))),
                                     np.hstack((y_z_iso_pi, np.ones_like(y_z_iso_pi))),
                                     np.hstack((y_e_iso_n, np.ones_like(y_e_iso_n) * 2)),
                                     np.hstack((y_z_iso_n, np.ones_like(y_z_iso_n) * 3))))

            y_test = np.hstack((y_test, np.zeros_like(y_test)))

        elif task == 'z_iso_pi':
            X_e_iso_pi, X_e_iso_n, X_z_iso_n, y_e_iso_pi, y_e_iso_n, y_z_iso_n = \
                test_leakage(X_e_iso_pi, X_e_iso_n, X_z_iso_n, y_e_iso_pi, y_e_iso_n, y_z_iso_n, X_test)
            # Augment the input with zeroes, ones, twos, threes to indicate the required output dimension
            X_augmented = np.vstack((np.append(X_e_iso_pi, np.zeros((len(X_e_iso_pi), 1)), axis=1),
                                     np.append(X_train, np.ones((len(X_train), 1)), axis=1),
                                     np.append(X_e_iso_n, np.ones((len(X_e_iso_n), 1)) * 2, axis=1),
                                     np.append(X_z_iso_n, np.ones((len(X_z_iso_n), 1)) * 3, axis=1)))

            X_test = np.append(X_test, np.ones((len(X_test), 1)), axis=1)
            X_train = np.append(X_train, np.ones((len(X_train), 1)), axis=1)

            # Augment the Y data with zeroes, ones, twos and threes that specify a likelihood from the list of likelihoods
            Y_augmented = np.vstack((np.hstack((y_e_iso_pi, np.zeros_like(y_e_iso_pi))),
                                     np.hstack((y_train, np.ones_like(y_train))),
                                     np.hstack((y_e_iso_n, np.ones_like(y_e_iso_n) * 2)),
                                     np.hstack((y_z_iso_n, np.ones_like(y_z_iso_n) * 3))))

            y_test = np.hstack((y_test, np.ones_like(y_test)))

        elif task == 'e_iso_n':
            X_e_iso_pi, X_z_iso_pi, X_z_iso_n, y_e_iso_pi, y_z_iso_pi, y_z_iso_n = \
                test_leakage(X_e_iso_pi, X_z_iso_pi, X_z_iso_n, y_e_iso_pi, y_z_iso_pi, y_z_iso_n, X_test)
            # Augment the input with zeroes, ones, twos, threes to indicate the required output dimension
            X_augmented = np.vstack((np.append(X_e_iso_pi, np.zeros((len(X_e_iso_pi), 1)), axis=1),
                                     np.append(X_z_iso_pi, np.ones((len(X_z_iso_pi), 1)), axis=1),
                                     np.append(X_train, np.ones((len(X_train), 1)) * 2, axis=1),
                                     np.append(X_z_iso_n, np.ones((len(X_z_iso_n), 1)) * 3, axis=1)))

            X_test = np.append(X_test, np.ones((len(X_test), 1)) * 2, axis=1)
            X_train = np.append(X_train, np.ones((len(X_train), 1)) * 2, axis=1)

            # Augment the Y data with zeroes, ones, twos and threes that specify a likelihood from the list of likelihoods
            Y_augmented = np.vstack((np.hstack((y_e_iso_pi, np.zeros_like(y_e_iso_pi))),
                                     np.hstack((y_z_iso_pi, np.ones_like(y_z_iso_pi))),
                                     np.hstack((y_train, np.ones_like(y_train) * 2)),
                                     np.hstack((y_z_iso_n, np.ones_like(y_z_iso_n) * 3))))

            y_test = np.hstack((y_test, np.ones_like(y_test) * 2))

        else:
            X_e_iso_pi, X_e_iso_n, X_z_iso_pi, y_e_iso_pi, y_e_iso_n, y_z_iso_pi = \
                test_leakage(X_e_iso_pi, X_e_iso_n, X_z_iso_pi, y_e_iso_pi, y_e_iso_n, y_z_iso_pi, X_test)
            # Augment the input with zeroes, ones, twos, threes to indicate the required output dimension
            X_augmented = np.vstack((np.append(X_e_iso_pi, np.zeros((len(X_e_iso_pi), 1)), axis=1),
                                     np.append(X_z_iso_pi, np.ones((len(X_z_iso_pi), 1)), axis=1),
                                     np.append(X_e_iso_n, np.ones((len(X_e_iso_n), 1)) * 2, axis=1),
                                     np.append(X_train, np.ones((len(X_train), 1)) * 3, axis=1)))

            X_test = np.append(X_test, np.ones((len(X_test), 1)) * 3, axis=1)
            X_train = np.append(X_train, np.ones((len(X_train), 1)) * 3, axis=1)

            # Augment the Y data with zeroes, ones, twos and threes that specify a likelihood from the list of likelihoods
            Y_augmented = np.vstack((np.hstack((y_e_iso_pi, np.zeros_like(y_e_iso_pi))),
                                     np.hstack((y_z_iso_pi, np.ones_like(y_z_iso_pi))),
                                     np.hstack((y_e_iso_n, np.ones_like(y_e_iso_n) * 2)),
                                     np.hstack((y_train, np.ones_like(y_train) * 3))))

            y_test = np.hstack((y_test, np.ones_like(y_test) * 3))

        # Base kernel
        k = Tanimoto(active_dims=tanimoto_active_dims)
        #set_trainable(k.variance, False)

        # Coregion kernel
        coreg = gpflow.kernels.Coregion(output_dim=output_dim, rank=rank, active_dims=[feature_dim])

        # Create product kernel
        kern = k * coreg

        # This likelihood switches between Gaussian noise with different variances for each f_i:
        lik = gpflow.likelihoods.SwitchedLikelihood([gpflow.likelihoods.Gaussian(), gpflow.likelihoods.Gaussian(),
                                                     gpflow.likelihoods.Gaussian(), gpflow.likelihoods.Gaussian()])

        # now build the GP model as normal
        m = gpflow.models.VGP((X_augmented, Y_augmented), mean_function=Constant(np.mean(y_train[:, 0])), kernel=kern, likelihood=lik)

        # fit the covariance function parameters
        maxiter = ci_niter(1000)
        gpflow.optimizers.Scipy().minimize(m.training_loss, m.trainable_variables, options=dict(maxiter=maxiter), method="L-BFGS-B",)
        print_summary(m)

        # mean and variance GP prediction

        y_pred, y_var = m.predict_f(X_test)

        # Output Standardised RMSE and RMSE on Train Set

        y_pred_train, _ = m.predict_f(X_train)
        train_rmse_stan = np.sqrt(mean_squared_error(y_train, y_pred_train))
        train_rmse = np.sqrt(mean_squared_error(y_train, y_pred_train))
        print("\nStandardised Train RMSE: {:.3f}".format(train_rmse_stan))
        print("Train RMSE: {:.3f}".format(train_rmse))

        score = r2_score(y_test[:, 0], y_pred)
        rmse = np.sqrt(mean_squared_error(y_test[:, 0], y_pred))
        mae = mean_absolute_error(y_test[:, 0], y_pred)

        print("\nR^2: {:.3f}".format(score))
        print("RMSE: {:.3f}".format(rmse))
        print("MAE: {:.3f}".format(mae))

        r2_list.append(score)
        rmse_list.append(rmse)
        mae_list.append(mae)

        B = coreg.output_covariance().numpy()
        print("B =", B)
        _ = plt.imshow(B)

    r2_list = np.array(r2_list)
    rmse_list = np.array(rmse_list)
    mae_list = np.array(mae_list)

    print("\nmean R^2: {:.4f} +- {:.4f}".format(np.mean(r2_list), np.std(r2_list)/np.sqrt(len(r2_list))))
    print("mean RMSE: {:.4f} +- {:.4f}".format(np.mean(rmse_list), np.std(rmse_list)/np.sqrt(len(rmse_list))))
    print("mean MAE: {:.4f} +- {:.4f}\n".format(np.mean(mae_list), np.std(mae_list)/np.sqrt(len(mae_list))))


if __name__ == '__main__':

    parser = argparse.ArgumentParser()

    parser.add_argument('-p', '--path', type=str, default='../dataset/photoswitches.csv',
                        help='Path to the photoswitches.csv file.')
    parser.add_argument('-t', '--task', type=str, default='z_iso_n',
                        help='str specifying the task. One of [e_iso_pi, z_iso_pi, e_iso_n, z_iso_n].')
    parser.add_argument('-r', '--representation', type=str, default='fragments',
                        help='str specifying the molecular representation. '
                             'One of [fingerprints, fragments, fragprints].')
    parser.add_argument('-pca', '--use_pca', type=bool, default=False,
                        help='If True apply PCA to perform Principal Components Regression.')
    parser.add_argument('-n', '--n_trials', type=int, default=20,
                        help='int specifying number of random train/test splits to use')
    parser.add_argument('-ts', '--test_set_size', type=float, default=0.2,
                        help='float in range [0, 1] specifying fraction of dataset to use as test set')

    args = parser.parse_args()

    main(args.path, args.task, args.representation, args.use_pca, args.n_trials, args.test_set_size)
