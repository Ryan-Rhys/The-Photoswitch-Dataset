# Copyright Ryan-Rhys Griffiths and Aditya Raymond Thawani 2020
# Author: Ryan-Rhys Griffiths
"""
Script for training a model to predict properties in the photoswitch dataset using Gaussian Process Regression.
"""

import gpflow
from gpflow.mean_functions import Constant
from gpflow.utilities import print_summary
from matplotlib import pyplot as plt
import numpy as np
from rdkit.Chem import AllChem, Descriptors, MolFromSmiles
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error

from data_utils import transform_data, DataLoader
from kernels import Tanimoto


PATH = '~/ml_physics/Photoswitches/dataset/photoswitches.csv'  # Change as appropriate
TASK = 'z_iso_n'  # ['thermal', 'e_iso_pi', 'z_iso_pi', 'e_iso_n', 'z_iso_n']
use_fragments = True  # If True use RDKit fragments as opposed to Morgan fingerprints
use_pca = False  # If True apply PCA to perform Principal Components Regression.
n_trials = 200  # number of random train/test splits to use
test_set_size = 0.2  # fraction of datapoints to use in the test set
use_rmse_conf = True  # Whether to use rmse confidence or mae confidence


if __name__ == '__main__':

    data_loader = DataLoader(TASK, PATH)
    smiles_list, y = data_loader.load_property_data()

    if not use_fragments:

        feat = 'fingerprints'

        rdkit_mols = [MolFromSmiles(smiles) for smiles in smiles_list]
        X = [AllChem.GetMorganFingerprintAsBitVect(mol, 3, nBits=2048) for mol in rdkit_mols]
        X = np.asarray(X)

    else:

        feat = 'fragments'

        # descList[115:] contains fragment-based features only
        # (https://www.rdkit.org/docs/source/rdkit.Chem.Fragments.html)

        fragments = {d[0]: d[1] for d in Descriptors.descList[115:]}
        X = np.zeros((len(smiles_list), len(fragments)))
        for i in range(len(smiles_list)):
            mol = MolFromSmiles(smiles_list[i])
            try:
                features = [fragments[d](mol) for d in fragments]
            except:
                raise Exception('molecule {}'.format(i) + ' is not canonicalised')
            X[i, :] = features

    # If True we perform Principal Components Regression

    if use_pca:
        n_components = 50
    else:
        n_components = None

    num_features = np.shape(X)[1]

    # We define the Gaussian Process Regression Model using the Tanimoto kernel

    m = None

    def objective_closure():
        return -m.log_marginal_likelihood()

    r2_list = []
    rmse_list = []
    mae_list = []

    # We pre-allocate arrays for plotting confidence-error curves

    _, _, _, y_test = train_test_split(X, y, test_size=test_set_size)  # To get test set size
    n_test = len(y_test)

    rmse_confidence_list = np.zeros((n_trials, n_test))
    mae_confidence_list = np.zeros((n_trials, n_test))

    print('\nBeginning training loop...')
    j = 0  # index for saving results

    for i in range(0, n_trials):

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_set_size, random_state=i)

        y_train = y_train.reshape(-1, 1)
        y_test = y_test.reshape(-1, 1)

        #  We standardise the outputs but leave the inputs unchanged

        _, y_train, _, y_test, y_scaler = transform_data(X_train, y_train, X_test, y_test, n_components=None)

        X_train = X_train.astype(np.float64)
        X_test = X_test.astype(np.float64)

        k = Tanimoto()
        #k = gpflow.kernels.RBF(lengthscales=np.ones(num_features))
        m = gpflow.models.GPR(data=(X_train, y_train), mean_function=Constant(np.mean(y_train)), kernel=k, noise_variance=1)

        # Optimise the kernel variance and noise level by the marginal likelihood

        opt = gpflow.optimizers.Scipy()
        opt.minimize(objective_closure, m.trainable_variables, options=dict(maxiter=100))
        print_summary(m)

        # mean and variance GP prediction

        y_pred, y_var = m.predict_f(X_test)
        y_pred = y_scaler.inverse_transform(y_pred)
        y_test = y_scaler.inverse_transform(y_test)

        # Compute scores for confidence curve plotting.

        ranked_confidence_list = np.argsort(y_var, axis=0).flatten()

        for k in range(len(y_test)):

            # Construct the RMSE error for each level of confidence

            conf = ranked_confidence_list[0:k+1]
            rmse = np.sqrt(mean_squared_error(y_test[conf], y_pred[conf]))
            rmse_confidence_list[i, k] = rmse

            # Construct the MAE error for each level of confidence

            mae = mean_absolute_error(y_test[conf], y_pred[conf])
            mae_confidence_list[i, k] = mae

        score = r2_score(y_test, y_pred)
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        mae = mean_absolute_error(y_test, y_pred)

        print("\nR^2: {:.3f}".format(score))
        print("RMSE: {:.3f}".format(rmse))
        print("MAE: {:.3f}".format(mae))

        r2_list.append(score)
        rmse_list.append(rmse)
        mae_list.append(mae)

        np.savetxt(TASK + '/results/gpr/_seed_'+str(j)+'_ypred_'+feat+'.txt', y_pred)
        np.savetxt(TASK + '/results/gpr/_seed_'+str(j)+'_ytest.txt', y_test)
        np.savetxt(TASK + '/results/gpr/_seed_'+str(j)+'_ystd_'+feat+'.txt', np.sqrt(y_var))

        j += 1

    r2_list = np.array(r2_list)
    rmse_list = np.array(rmse_list)
    mae_list = np.array(mae_list)

    print("\nmean R^2: {:.4f} +- {:.4f}".format(np.mean(r2_list), np.std(r2_list)/np.sqrt(len(r2_list))))
    print("mean RMSE: {:.4f} +- {:.4f}".format(np.mean(rmse_list), np.std(rmse_list)/np.sqrt(len(rmse_list))))
    print("mean MAE: {:.4f} +- {:.4f}\n".format(np.mean(mae_list), np.std(mae_list)/np.sqrt(len(mae_list))))

    # Plot confidence-error curves

    confidence_percentiles = np.arange(1e-14, 100, 100/len(y_test))  # 1e-14 instead of 0 to stop weirdness with len(y_test) = 29

    if use_rmse_conf:

        rmse_mean = np.mean(rmse_confidence_list, axis=0)
        rmse_std = np.std(rmse_confidence_list, axis=0)

        # We flip because we want the most confident predictions on the right-hand side of the plot

        rmse_mean = np.flip(rmse_mean)
        rmse_std = np.flip(rmse_std)

        # One-sigma error bars

        lower = rmse_mean - rmse_std
        upper = rmse_mean + rmse_std

        plt.plot(confidence_percentiles, rmse_mean, label='mean')
        plt.fill_between(confidence_percentiles, lower, upper, alpha=0.2)
        plt.xlabel('Confidence Percentile')
        plt.ylabel('RMSE (nm)')
        plt.ylim([0, np.max(upper) + 1])
        plt.xlim([0, 100*((len(y_test) - 1) / len(y_test))])
        plt.yticks(np.arange(0, np.max(upper) + 1, 5.0))
        plt.savefig(TASK + '/results/gpr/confidence_curve_rmse.png')
        plt.show()

    else:

        # We plot the Mean-absolute error confidence-error curves

        mae_mean = np.mean(mae_confidence_list, axis=0)
        mae_std = np.std(mae_confidence_list, axis=0)

        mae_mean = np.flip(mae_mean)
        mae_std = np.flip(mae_std)

        lower = mae_mean - mae_std
        upper = mae_mean + mae_std

        plt.plot(confidence_percentiles, mae_mean, label='mean')
        plt.fill_between(confidence_percentiles, lower, upper, alpha=0.2)
        plt.xlabel('Confidence Percentile')
        plt.ylabel('MAE (nm)')
        plt.ylim([0, np.max(upper) + 1])
        plt.xlim([0, 100 * ((len(y_test) - 1) / len(y_test))])
        plt.yticks(np.arange(0, np.max(upper) + 1, 5.0))
        plt.savefig(TASK + '/results/gpr/confidence_curve_mae.png')
        plt.show()
