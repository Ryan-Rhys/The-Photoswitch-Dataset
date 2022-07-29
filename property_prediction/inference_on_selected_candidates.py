"""
Candidate predictions for the MOGP model.
Author: Ryan-Rhys Griffiths
"""

import gpflow
from gpflow.ci_utils import ci_niter
from gpflow.mean_functions import Constant
from gpflow.utilities import print_summary
import numpy as np

from data_utils import TaskDataLoader, featurise_mols
from kernels import Tanimoto

representation = 'fragprints'
task = 'z_iso_pi'
path = '../dataset/photoswitches.csv'

if __name__ == '__main__':

    candidate_list = ['CCOC1=CC=C(/N=N/C2=C3N=CC(C#N)=C(N)N3N=C2N)C=C1',
                      'NC1=NN2C(N)=C(C#N)C=NC2=C1/N=N/C3=CC4=C(OCO4)C=C3',
                      'COC1=CC=C(C2=NC3=C(C(N)=NN3C(N)=C2C#N)/N=N/C4=CC5=C(OCO5)C=C4)C=C1',
                      'O=S(C1=CC=C(/N=N/C2=C3N=C(C4=CC=CS4)C(C#N)=C(N)N3N=C2N)C=C1)(N)=O',
                      'CSC1=C(C(N)=NC2=C(C(N)=NN12)/N=N/C3=CC=C(S(N)(=O)=O)C=C3)C#N',
                      'COC1=CC=C(/N=N/C2=C3N=C(C4=CC=CC=C4)C(C#N)=C(N)N3N=C2N)C=C1',
                      '[O-]Cl(=O)(=O)=O.CCN(C1=CC=C(/N=N/C2=C([N+]([O-])=O)C(C)=[N+](N2C)C)C=C1)CC',
                      'CN(C1=CC=C(/N=N/C2=NC(C#N)=C(C#N)N2)C=C1)C',
                      'CN1C(/N=N/C2=CC=C(NC3=CC=CC=C3)C=C2)=NC4=CC=CC=C14',
                      'CCN(S(=O)(C1=CC=C(/N=N/C2=C(C3=CC=CC=C3)N=C(N)S2)C=C1)=O)CC',
                      'O=C1N(C2=CC=CC=C2)N(C)C(C)=C1/N=N/C3=CC=C(N)C(OC)=C3']

    X_test = featurise_mols(candidate_list, representation)

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

    if task == 'e_iso_pi':
        X_train = X_e_iso_pi
        y_train = y_e_iso_pi
    elif task == 'z_iso_pi':
        X_train = X_z_iso_pi
        y_train = y_z_iso_pi
    elif task == 'e_iso_n':
        X_train = X_e_iso_n
        y_train = y_e_iso_n
    else:
        X_train = X_z_iso_n
        y_train = y_z_iso_n

    if task == 'e_iso_pi':
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

    elif task == 'z_iso_pi':
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

    elif task == 'e_iso_n':
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

    else:
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

    # Fit GP

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

    print(f'GP {representation} prediction is ')
    print(y_pred)
    print(f'GP {representation} variance is')
    print(y_var)

    np.savetxt(f'predictions/candidate_predictions/purchasable_multioutput_gp_task_{task}.txt', y_pred)
