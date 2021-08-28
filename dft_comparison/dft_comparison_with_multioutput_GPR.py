# Author: Ryan-Rhys Griffiths
"""
Property prediction comparison against DFT error. 99 molecules with DFT-computed values at the CAM-B3LYP level of
theory and 114 molecules with DFT-computed values at the PBE0 level of theory.
"""

import argparse

import gpflow
from gpflow.ci_utils import ci_niter
from gpflow.mean_functions import Constant
from gpflow.utilities import print_summary
import numpy as np
from sklearn.metrics import mean_squared_error

from data_utils import TaskDataLoader, featurise_mols
from kernels import Tanimoto


def main(path, path_to_dft_dataset, representation, theory_level):
    """
    :param path: str specifying path to photoswitches.csv file.
    :param path_to_dft_dataset: str specifying path to dft_comparison.csv file.
    :param representation: str specifying the molecular representation. One of ['fingerprints, 'fragments', 'fragprints']
    :param theory_level: str giving the level of theory to compare against - CAM-B3LYP or PBE0 ['CAM-B3LYP', 'PBE0']
    """

    task = 'e_iso_pi'  # e_iso_pi only task supported for TD-DFT comparison
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

    # Load in the other property values for multitask learning. e_iso_pi is a always the task in this instance.

    data_loader_z_iso_pi = TaskDataLoader('z_iso_pi', path)
    data_loader_e_iso_n = TaskDataLoader('e_iso_n', path)
    data_loader_z_iso_n = TaskDataLoader('z_iso_n', path)

    smiles_list_z_iso_pi, y_z_iso_pi = data_loader_z_iso_pi.load_property_data()
    smiles_list_e_iso_n, y_e_iso_n = data_loader_e_iso_n.load_property_data()
    smiles_list_z_iso_n, y_z_iso_n = data_loader_z_iso_n.load_property_data()

    y_z_iso_pi = y_z_iso_pi.reshape(-1, 1)
    y_e_iso_n = y_e_iso_n.reshape(-1, 1)
    y_z_iso_n = y_z_iso_n.reshape(-1, 1)

    X_z_iso_pi = featurise_mols(smiles_list_z_iso_pi, representation)
    X_e_iso_n = featurise_mols(smiles_list_e_iso_n, representation)
    X_z_iso_n = featurise_mols(smiles_list_z_iso_n, representation)

    output_dim = 4  # Number of outputs
    rank = 1  # Rank of W
    feature_dim = len(X_no_dft[0, :])

    tanimoto_active_dims = [i for i in range(feature_dim)]  # active dims for Tanimoto base kernel.

    mae_list = []
    dft_mae_list = []

    # We define the Gaussian Process optimisation objective

    m = None

    def objective_closure():
        return -m.log_marginal_likelihood()

    print('\nBeginning training loop...')

    for i in range(len(y_with_dft)):

        X_train = np.delete(X_with_dft, i, axis=0)
        y_train = np.delete(y_with_dft, i)
        X_test = X_with_dft[i].reshape(1, -1)
        y_test = y_with_dft[i]
        dft_test = dft_vals[i]

        X_train = np.concatenate((X_train, X_no_dft))
        y_train = np.concatenate((y_train, y_no_dft))
        y_train = y_train.reshape(-1, 1)
        y_test = y_test.reshape(-1, 1)

        X_train = X_train.astype(np.float64)
        X_test = X_test.astype(np.float64)

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

        # Output Standardised RMSE and RMSE on Train Set

        y_pred_train, _ = m.predict_f(X_train)
        train_rmse_stan = np.sqrt(mean_squared_error(y_train, y_pred_train))
        train_rmse = np.sqrt(mean_squared_error(y_train, y_pred_train))
        print("\nStandardised Train RMSE: {:.3f}".format(train_rmse_stan))
        print("Train RMSE: {:.3f}".format(train_rmse))

        # mean and variance GP prediction

        y_pred, y_var = m.predict_f(X_test)

        # Output MAE for this trial

        mae = abs(y_test[:, 0] - y_pred)

        print("MAE: {}".format(mae))

        # Store values in order to compute the mean and standard error of the statistics across trials

        mae_list.append(mae)

        # DFT prediction scores on the same trial

        dft_mae = abs(y_test[:, 0] - dft_test)

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
    parser.add_argument('-r', '--representation', type=str, default='fragprints',
                        help='str specifying the molecular representation. '
                             'One of [fingerprints, fragments, fragprints].')
    parser.add_argument('-th', '--theory_level', type=str, default='PBE0',
                        help='level of theory to compare against - CAM-B3LYP or PBE0 [CAM-B3LYP, PBE0]')

    args = parser.parse_args()

    main(args.path, args.path_to_dft_dataset, args.representation, args.theory_level)
