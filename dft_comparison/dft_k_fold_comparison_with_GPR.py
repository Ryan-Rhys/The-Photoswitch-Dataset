# Author: Ryan-Rhys Griffiths
"""
Property prediction comparison against DFT error. 99 molecules with DFT-computed values at the CAM-B3LYP level of
theory and 114 molecules with DFT-computed values at the PBE0 level of theory.
"""

import argparse

import gpflow
from gpflow.mean_functions import Constant
from gpflow.utilities import print_summary
import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.model_selection import train_test_split

from data_utils import transform_data, TaskDataLoader, featurise_mols
from kernels import Tanimoto


def main(path, path_to_dft_dataset, task, representation, theory_level):
    """
    :param path: str specifying path to photoswitches.csv file.
    :param path_to_dft_dataset: str specifying path to dft_comparison.csv file.
    :param task: str specifying the task. e_iso_pi only supported task for the TD-DFT comparison.
    :param representation: str specifying the molecular representation. One of ['fingerprints, 'fragments', 'fragprints']
    :param theory_level: str giving the level of theory to compare against - CAM-B3LYP or PBE0 ['CAM-B3LYP', 'PBE0']
    """

    data_loader = TaskDataLoader(task, path)
    smiles_list, _, pbe0_vals, cam_vals, experimental_vals = data_loader.load_dft_comparison_data(path_to_dft_dataset)

    X = featurise_mols(smiles_list, representation)

    # Keep only non-duplicate entries because we're not considering effects of solvent

    non_duplicate_indices = np.array([i for i, smiles in enumerate(smiles_list) if smiles not in smiles_list[:i]])
    X = X[non_duplicate_indices, :]
    experimental_vals = experimental_vals[non_duplicate_indices]
    non_dup_pbe0 = np.array([i for i, smiles in enumerate(smiles_list) if smiles not in smiles_list[:i]])
    non_dup_cam = np.array([i for i, smiles in enumerate(smiles_list) if smiles not in smiles_list[:i]])
    pbe0_vals = pbe0_vals[non_dup_pbe0]
    cam_vals = cam_vals[non_dup_cam]

    # molecules with dft values to be split into train/test
    if theory_level == 'CAM-B3LYP':
        X_with_dft = np.delete(X, np.argwhere(np.isnan(cam_vals)), axis=0)
        y_with_dft = np.delete(experimental_vals, np.argwhere(np.isnan(cam_vals)))
        # DFT values for the CAM-B3LYP level of theory
        dft_vals = np.delete(cam_vals, np.argwhere(np.isnan(cam_vals)))
        # molecules with no dft vals must go into the training set.
        X_no_dft = np.delete(X, np.argwhere(~np.isnan(cam_vals)), axis=0)
        y_no_dft = np.delete(experimental_vals, np.argwhere(~np.isnan(cam_vals)))
    else:
        X_with_dft = np.delete(X, np.argwhere(np.isnan(pbe0_vals)), axis=0)
        y_with_dft = np.delete(experimental_vals, np.argwhere(np.isnan(pbe0_vals)))
        # DFT values for the PBE0 level of theory
        dft_vals = np.delete(pbe0_vals, np.argwhere(np.isnan(pbe0_vals)))
        # molecules with no dft vals must go into the training set.
        X_no_dft = np.delete(X, np.argwhere(~np.isnan(pbe0_vals)), axis=0)
        y_no_dft = np.delete(experimental_vals, np.argwhere(~np.isnan(pbe0_vals)))

    mae_list = []
    dft_mae_list = []

    # We define the Gaussian Process optimisation objective

    m = None

    def objective_closure():
        return -m.log_marginal_likelihood()

    print('\nBeginning training loop...')

    for i in range(5):

        X_train, X_test, y_train, y_test = train_test_split(X_with_dft, y_with_dft, test_size=0.5, random_state=i)
        X_dud, _, _, dft_test = train_test_split(X_with_dft, dft_vals, test_size=0.6, random_state=i)

        X_train = np.concatenate((X_train, X_no_dft))
        y_train = np.concatenate((y_train, y_no_dft))
        y_train = y_train.reshape(-1, 1)
        y_test = y_test.reshape(-1, 1)

        #  We standardise the outputs but leave the inputs unchanged

        _, y_train, _, y_test, y_scaler = transform_data(X_train, y_train, X_test, y_test)

        X_train = X_train.astype(np.float64)
        X_test = X_test.astype(np.float64)

        k = Tanimoto()
        m = gpflow.models.GPR(data=(X_train, y_train), mean_function=Constant(np.mean(y_train)), kernel=k, noise_variance=1)

        # Optimise the kernel variance and noise level by the marginal likelihood

        opt = gpflow.optimizers.Scipy()
        opt.minimize(objective_closure, m.trainable_variables, options=dict(maxiter=100))
        print_summary(m)

        # Output Standardised RMSE and RMSE on Train Set

        y_pred_train, _ = m.predict_f(X_train)
        train_rmse_stan = np.sqrt(mean_squared_error(y_train, y_pred_train))
        train_rmse = np.sqrt(mean_squared_error(y_scaler.inverse_transform(y_train), y_scaler.inverse_transform(y_pred_train)))
        print("\nStandardised Train RMSE: {:.3f}".format(train_rmse_stan))
        print("Train RMSE: {:.3f}".format(train_rmse))

        # mean and variance GP prediction

        y_pred, y_var = m.predict_f(X_test)
        y_pred = y_scaler.inverse_transform(y_pred)
        y_test = y_scaler.inverse_transform(y_test)

        # Output MAE for this trial

        mae = mean_absolute_error(y_test, y_pred)

        print("MAE: {}".format(mae))

        # Store values in order to compute the mean and standard error of the statistics across trials

        mae_list.append(mae)

        # DFT prediction scores on the same trial

        dft_mae = mean_absolute_error(y_test, dft_test)

        dft_mae_list.append(dft_mae)

    mae_list = np.array(mae_list)
    dft_mae_list = np.array(dft_mae_list)

    print("\nmean GP-Tanimoto MAE: {:.4f} +- {:.4f}\n".format(np.mean(mae_list), np.std(mae_list)/np.sqrt(len(mae_list))))

    print("mean {} MAE: {:.4f} +- {:.4f}\n".format(theory_level, np.mean(dft_mae_list), np.std(dft_mae_list)/np.sqrt(len(dft_mae_list))))


if __name__ == '__main__':

    parser = argparse.ArgumentParser()

    parser.add_argument('-p', '--path', type=str, default='../dataset/photoswitches.csv',
                        help='Path to the photoswitches.csv file.')
    parser.add_argument('-pd', '--path_to_dft_dataset', type=str, default='../dataset/dft_comparison.csv',
                        help='str giving path to dft_comparison.csv file')
    parser.add_argument('-t', '--task', type=str, default='e_iso_pi',
                        help='str specifying the task. e_iso_pi only task supported for the TD-DFT comparison.')
    parser.add_argument('-r', '--representation', type=str, default='fragprints',
                        help='str specifying the molecular representation. '
                             'One of [fingerprints, fragments, fragprints].')
    parser.add_argument('-th', '--theory_level', type=str, default='CAM-B3LYP',
                        help='level of theory to compare against - CAM-B3LYP or PBE0 [CAM-B3LYP, PBE0]')

    args = parser.parse_args()

    main(args.path, args.path_to_dft_dataset, args.task, args.representation, args.theory_level)
