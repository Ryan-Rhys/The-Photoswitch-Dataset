# Author: Ryan-Rhys Griffiths
"""
Property prediction comparison against DFT error as of 99 molecules with DFT-computed values at the CAM-B3LYP level of
theory and 114 molecules with DFT-computed values at the PBE0 level of theory.
"""

import gpflow
from gpflow.mean_functions import Constant
from gpflow.utilities import print_summary
import numpy as np
from sklearn.metrics import mean_squared_error

from data_utils import transform_data, TaskDataLoader, featurise_mols
from kernels import Tanimoto

PATH = '../dataset/photoswitches.csv'  # Change as appropriate
DFT_PATH = '../dataset/dft_comparison.csv'  # Change as appropriate
TASK = 'e_iso_pi'  # e_iso_pi only task supported for DFT comparison
representation = 'fragprints'  # ['fingerprints, 'fragments', 'fragprints']
test_set_size = 0.2
theory_level = 'CAM-B3LYP'  # level of theory to compare against - CAM-B3LYP or PBE0 ['CAM-B3LYP', 'PBE0']


if __name__ == '__main__':

    data_loader = TaskDataLoader(TASK, PATH)
    smiles_list, _, pbe0_vals, cam_vals, experimental_vals = data_loader.load_dft_comparison_data(DFT_PATH)

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

        mae = abs(y_test - y_pred)

        print("MAE: {}".format(mae))

        # Store values in order to compute the mean and standard error of the statistics across trials

        mae_list.append(mae)

        # DFT prediction scores on the same trial

        dft_mae = abs(y_test - dft_test)

        dft_mae_list.append(dft_mae)

    mae_list = np.array(mae_list)
    dft_mae_list = np.array(dft_mae_list)

    print("\nmean GP-Tanimoto MAE: {:.4f} +- {:.4f}\n".format(np.mean(mae_list), np.std(mae_list)/np.sqrt(len(mae_list))))

    print("mean {} MAE: {:.4f} +- {:.4f}\n".format(theory_level, np.mean(dft_mae_list), np.std(dft_mae_list)/np.sqrt(len(dft_mae_list))))
