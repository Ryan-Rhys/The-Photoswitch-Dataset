# Copyright Ryan-Rhys Griffiths 2020
# Author: Ryan-Rhys Griffiths
"""
Script for comparing against human performance on a set of 5 molecules with Tanimoto GP.
"""

import argparse

import gpflow
from gpflow.mean_functions import Constant
from gpflow.utilities import print_summary
import numpy as np
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error

from data_utils import transform_data, TaskDataLoader, featurise_mols
from kernels import Tanimoto


def main(path, representation):
    """
    :param path: str specifying path to dataset.
    :param representation: str specifying the molecular representation. One of ['fingerprints, 'fragments', 'fragprints']
    """

    task = 'e_iso_pi'  # Always e_iso_pi for human performance comparison
    data_loader = TaskDataLoader(task, path)
    smiles_list, y = data_loader.load_property_data()
    X = featurise_mols(smiles_list, representation)

    # 5 test molecules

    test_smiles = ['BrC1=CC=C(/N=N/C2=CC=CC=C2)C=C1',
                   'O=[N+]([O-])C1=CC=C(/N=N/C2=CC=CC=C2)C=C1',
                   'CC(C=C1)=CC=C1/N=N/C2=CC=C(N(C)C)C=C2',
                   'BrC1=CC([N+]([O-])=O)=CC([N+]([O-])=O)=C1/N=N/C2=CC([H])=C(C=C2[H])N(CC)CC',
                   'ClC%11=CC([N+]([O-])=O)=CC(C#N)=C%11/N=N/C%12=CC([H])=C(C=C%12OC)N(CC)CC']

    # and their indices in the loaded data
    test_smiles_indices = [116, 131, 168, 221, 229]

    X_train = np.delete(X, np.array(test_smiles_indices), axis=0)
    y_train = np.delete(y, np.array(test_smiles_indices))
    X_test = X[[116, 131, 168, 221, 229]]

    # experimental wavelength values in EtOH. Main csv file has 400nm instead of 407nm because measurement was
    # under a different solvent
    y_test = y[[116, 131, 168, 221, 229]]
    y_test[2] = 407.

    y_train = y_train.reshape(-1, 1)
    y_test = y_test.reshape(-1, 1)

    #  We standardise the outputs but leave the inputs unchanged

    _, y_train, _, y_test, y_scaler = transform_data(X_train, y_train, X_test, y_test)

    X_train = X_train.astype(np.float64)
    X_test = X_test.astype(np.float64)

    num_features = np.shape(X)[1]

    # We define the Gaussian Process Regression Model using the Tanimoto kernel

    m = None

    def objective_closure():
        return -m.log_marginal_likelihood()

    # for plotting confidence-error curves

    rmse_confidence_list = []
    mae_confidence_list = []

    k = Tanimoto()
    m = gpflow.models.GPR(data=(X_train, y_train), mean_function=Constant(np.mean(y_train)), kernel=k, noise_variance=1)

    # Optimise the kernel variance and noise level by the marginal likelihood

    opt = gpflow.optimizers.Scipy()
    opt.minimize(objective_closure, m.trainable_variables, options=dict(maxiter=100))
    print_summary(m)

    # mean and variance GP prediction

    y_pred, y_var = m.predict_f(X_test)
    y_pred = y_scaler.inverse_transform(y_pred)
    y_test = y_scaler.inverse_transform(y_test)

    # Output Standardised RMSE and RMSE on Train Set

    y_pred_train, _ = m.predict_f(X_train)
    train_rmse_stan = np.sqrt(mean_squared_error(y_train, y_pred_train))
    train_rmse = np.sqrt(mean_squared_error(y_scaler.inverse_transform(y_train), y_scaler.inverse_transform(y_pred_train)))
    print("\nStandardised Train RMSE: {:.3f}".format(train_rmse_stan))
    print("Train RMSE: {:.3f}".format(train_rmse))

    r2 = r2_score(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    mae = mean_absolute_error(y_test, y_pred)
    per_molecule = abs(y_pred - y_test)

    print("\n Averaged test statistics are")
    print("\nR^2: {:.3f}".format(r2))
    print("RMSE: {:.3f}".format(rmse))
    print("MAE: {:.3f}".format(mae))
    print("\nAbsolute error per molecule is {} ".format(per_molecule))


if __name__ == '__main__':

    parser = argparse.ArgumentParser()

    parser.add_argument('-p', '--path', type=str, default='../dataset/photoswitches.csv',
                        help='Path to the photoswitches.csv file.')
    parser.add_argument('-r', '--representation', type=str, default='fragprints',
                        help='str specifying the molecular representation. '
                             'One of [fingerprints, fragments, fragprints].')

    args = parser.parse_args()

    main(args.path, args.representation)
