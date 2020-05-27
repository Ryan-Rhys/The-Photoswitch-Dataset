# Author: Ryan-Rhys Griffiths
"""
Property prediction comparison against DFT error as of 24th May 2020 there are 141 molecules with DFT-computed values
of the E isomer pi-pi* transition wavelength for both the PBE0 and CAM-B3LYP levels of theory.
"""

import gpflow
from gpflow.mean_functions import Constant
from gpflow.utilities import print_summary
import numpy as np
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from sklearn.model_selection import train_test_split

from data_utils import transform_data, TaskDataLoader, featurise_mols
from kernels import Tanimoto

PATH = '~/ml_physics/Photoswitches/dataset/photoswitches.csv'  # Change as appropriate
DFT_PATH = '~/ml_physics/Photoswitches/dataset/dft_comparison.csv'  # Change as appropriate
TASK = 'e_iso_pi'  # e_iso_pi only task supported for DFT comparison
use_pca = False
representation = 'fragprints'  # ['fingerprints, 'fragments', 'fragprints']
n_trials = 200
test_set_size = 0.2


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

    # molecules with no dft vals must go into the training set.
    X_no_dft = np.delete(X, np.argwhere(~np.isnan(cam_vals)), axis=0)
    y_no_dft = np.delete(experimental_vals, np.argwhere(~np.isnan(cam_vals)))

    # molecules with dft values to be split into train/test
    X_with_dft = np.delete(X, np.argwhere(np.isnan(cam_vals)), axis=0)
    y_with_dft = np.delete(experimental_vals, np.argwhere(np.isnan(cam_vals)))

    # DFT values for the CAM-B3LYP and PBE0 levels of theory
    cam_vals = np.delete(cam_vals, np.argwhere(np.isnan(cam_vals)))
    pbe0_vals = np.delete(pbe0_vals, np.argwhere(np.isnan(pbe0_vals)))

    r2_list, rmse_list, mae_list = [], [], []
    cam_r2_list, cam_rmse_list, cam_mae_list = [], [], []
    pbe0_r2_list, pbe0_rmse_list, pbe0_mae_list = [], [], []

    # We define the Gaussian Process optimisation objective

    m = None

    def objective_closure():
        return -m.log_marginal_likelihood()

    print('\nBeginning training loop...')

    for i in range(n_trials):

        X_train, X_test, y_train, y_test = train_test_split(X_with_dft, y_with_dft, test_size=test_set_size, random_state=i)

        # Apply same split to the CAM-B3LYP and PBE0-computed values.
        _, _, _, cam_dft = train_test_split(X_with_dft, cam_vals, test_size=test_set_size, random_state=i)
        _, _, _, pbe0_dft = train_test_split(X_with_dft, pbe0_vals, test_size=test_set_size, random_state=i)

        X_train = np.concatenate((X_train, X_no_dft))
        y_train = np.concatenate((y_train, y_no_dft))
        y_train = y_train.reshape(-1, 1)
        y_test = y_test.reshape(-1, 1)

        #  We standardise the outputs but leave the inputs unchanged

        _, y_train, _, y_test, y_scaler = transform_data(X_train, y_train, X_test, y_test, n_components=None)

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

        # Output R^2, RMSE and MAE for this trial

        score = r2_score(y_test, y_pred)
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        mae = mean_absolute_error(y_test, y_pred)

        print("\nR^2: {:.3f}".format(score))
        print("RMSE: {:.3f}".format(rmse))
        print("MAE: {:.3f}".format(mae))

        # Store values in order to compute the mean and standard error of the statistics across trials

        r2_list.append(score)
        rmse_list.append(rmse)
        mae_list.append(mae)

        # DFT prediction scores on the same trial

        cam_score = r2_score(y_test, cam_dft)
        cam_rmse = np.sqrt(mean_squared_error(y_test, cam_dft))
        cam_mae = mean_absolute_error(y_test, cam_dft)

        cam_r2_list.append(cam_score)
        cam_rmse_list.append(cam_rmse)
        cam_mae_list.append(cam_mae)

        pbe0_score = r2_score(y_test, pbe0_dft)
        pbe0_rmse = np.sqrt(mean_squared_error(y_test, pbe0_dft))
        pbe0_mae = mean_absolute_error(y_test, pbe0_dft)

        pbe0_r2_list.append(pbe0_score)
        pbe0_rmse_list.append(pbe0_rmse)
        pbe0_mae_list.append(pbe0_mae)

    r2_list, rmse_list, mae_list = np.array(r2_list), np.array(rmse_list), np.array(mae_list)
    cam_r2_list, cam_rmse_list, cam_mae_list = np.array(cam_r2_list), np.array(cam_rmse_list), np.array(cam_mae_list)
    pbe0_r2_list, pbe0_rmse_list, pbe0_mae_list = np.array(pbe0_r2_list), np.array(pbe0_rmse_list), np.array(pbe0_mae_list)

    print("\nmean R^2: {:.4f} +- {:.4f}".format(np.mean(r2_list), np.std(r2_list)/np.sqrt(len(r2_list))))
    print("mean RMSE: {:.4f} +- {:.4f}".format(np.mean(rmse_list), np.std(rmse_list)/np.sqrt(len(rmse_list))))
    print("mean MAE: {:.4f} +- {:.4f}\n".format(np.mean(mae_list), np.std(mae_list)/np.sqrt(len(mae_list))))

    print("\nmean CAM-B3LYP R^2: {:.4f} +- {:.4f}".format(np.mean(cam_r2_list), np.std(cam_r2_list)/np.sqrt(len(cam_r2_list))))
    print("mean CAM-B3LYP RMSE: {:.4f} +- {:.4f}".format(np.mean(cam_rmse_list), np.std(cam_rmse_list)/np.sqrt(len(cam_rmse_list))))
    print("mean CAM-B3LYP MAE: {:.4f} +- {:.4f}\n".format(np.mean(cam_mae_list), np.std(cam_mae_list)/np.sqrt(len(cam_mae_list))))

    print("\nmean PBE0 R^2: {:.4f} +- {:.4f}".format(np.mean(pbe0_r2_list), np.std(pbe0_r2_list)/np.sqrt(len(pbe0_r2_list))))
    print("mean PBE0 RMSE: {:.4f} +- {:.4f}".format(np.mean(pbe0_rmse_list), np.std(pbe0_rmse_list)/np.sqrt(len(pbe0_rmse_list))))
    print("mean PBE0 MAE: {:.4f} +- {:.4f}\n".format(np.mean(pbe0_mae_list), np.std(pbe0_mae_list)/np.sqrt(len(pbe0_mae_list))))
