# Copyright Ryan-Rhys Griffiths 2020
# Author: Ryan-Rhys Griffiths
"""
Script for comparing against human performance on a set of 5 molecules with Tanimoto MOGP.
"""

import argparse

import gpflow
from gpflow.ci_utils import ci_niter
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

    task = 'e_iso_pi'  # task always e_iso_pi with human performance comparison
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

    # #  We standardise the outputs but leave the inputs unchanged
    #
    # _, y_train, _, y_test, y_scaler = transform_data(X_train, y_train, X_test, y_test)

    X_train = X_train.astype(np.float64)
    X_test = X_test.astype(np.float64)

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
    feature_dim = len(X_train[0, :])

    tanimoto_active_dims = [i for i in range(feature_dim)]  # active dims for Tanimoto base kernel.

    # We define the Gaussian Process Regression Model using the Tanimoto kernel

    m = None

    def objective_closure():
        return -m.log_marginal_likelihood()

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
    # set_trainable(k.variance, False)

    # Coregion kernel
    coreg = gpflow.kernels.Coregion(output_dim=output_dim, rank=rank, active_dims=[feature_dim])

    # Create product kernel
    kern = k * coreg

    # This likelihood switches between Gaussian noise with different variances for each f_i:
    lik = gpflow.likelihoods.SwitchedLikelihood([gpflow.likelihoods.Gaussian(), gpflow.likelihoods.Gaussian(),
                                                 gpflow.likelihoods.Gaussian(), gpflow.likelihoods.Gaussian()])

    # now build the GP model as normal
    m = gpflow.models.VGP((X_augmented, Y_augmented), mean_function=Constant(np.mean(y_train[:, 0])), kernel=kern,
                          likelihood=lik)

    # fit the covariance function parameters
    maxiter = ci_niter(1000)
    gpflow.optimizers.Scipy().minimize(m.training_loss, m.trainable_variables, options=dict(maxiter=maxiter),
                                       method="L-BFGS-B", )
    print_summary(m)

    # mean and variance GP prediction

    y_pred, y_var = m.predict_f(X_test)

    # Output Standardised RMSE and RMSE on Train Set

    y_pred_train, _ = m.predict_f(X_train)
    train_rmse_stan = np.sqrt(mean_squared_error(y_train, y_pred_train))
    train_rmse = np.sqrt(mean_squared_error(y_train, y_pred_train))
    print("\nStandardised Train RMSE: {:.3f}".format(train_rmse_stan))
    print("Train RMSE: {:.3f}".format(train_rmse))

    r2 = r2_score(y_test[:, 0], y_pred)
    rmse = np.sqrt(mean_squared_error(y_test[:, 0], y_pred))
    mae = mean_absolute_error(y_test[:, 0], y_pred)
    per_molecule = np.diag(abs(y_pred - y_test[:, 0]))

    print("\n Averaged test statistics are")
    print("\nR^2: {:.3f}".format(r2))
    print("RMSE: {:.3f}".format(rmse))
    print("MAE: {:.3f}".format(mae))
    print("\nAbsolute error per molecule is {} ".format(per_molecule))
    print(f"predicted values are {y_pred}")
    print(f"true values are {y_test}")


if __name__ == '__main__':

    parser = argparse.ArgumentParser()

    parser.add_argument('-p', '--path', type=str, default='../dataset/photoswitches.csv',
                        help='Path to the photoswitches.csv file.')
    parser.add_argument('-r', '--representation', type=str, default='fragprints',
                        help='str specifying the molecular representation. '
                             'One of [fingerprints, fragments, fragprints].')

    args = parser.parse_args()

    main(args.path, args.representation)