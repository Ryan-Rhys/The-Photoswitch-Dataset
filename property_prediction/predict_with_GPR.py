"""
Script for training a model to predict properties in the photoswitch dataset using Gaussian Process Regression.
"""

import gpflow
from gpflow.utilities import print_summary
import numpy as np
from rdkit.Chem import AllChem, Descriptors, MolFromSmiles
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error

from data_utils import load_e_iso_pi_data, load_thermal_data, load_z_iso_pi_data, transform_data

PATH = '~/ml_physics/Photoswitches/dataset/photoswitches.csv'
TASK = 'e_iso_pi'  # ['thermal', 'e_iso_pi', 'z_iso_pi']
use_fragments = True
use_pca = False


if __name__ == '__main__':

    if TASK == 'thermal':
        smiles_list, y = load_thermal_data(PATH)
    elif TASK == 'e_iso_pi':
        smiles_list, y = load_e_iso_pi_data(PATH)
    elif TASK == 'z_iso_pi':
        smiles_list, y = load_z_iso_pi_data(PATH)
    else:
        raise Exception('Must specify a valid task')

    if not use_fragments:

        feat = 'fingerprints'

        if use_pca:
            n_components = 50

        rdkit_mols = [MolFromSmiles(smiles) for smiles in smiles_list]
        X = [AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=512) for mol in rdkit_mols]
        X = np.asarray(X)

    else:

        feat = 'fragments'

        if use_pca:
            n_components = 50

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

    num_features = np.shape(X)[1]

    m = None

    def objective_closure():
        return -m.log_marginal_likelihood()

    r2_list = []
    rmse_list = []
    mae_list = []

    print('\nBeginning training loop...')
    j = 0  # index for saving results

    for i in range(0, 25):

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=i)
        X_train, y_train, X_test, y_test, y_scaler = transform_data(X_train, y_train, X_test, y_test, n_components=None)

        k = gpflow.kernels.RBF(lengthscale=np.ones(num_features))
        m = gpflow.models.GPR(data=(X_train, y_train), kernel=k, noise_variance=1)

        opt = gpflow.optimizers.Scipy()

        opt_logs = opt.minimize(objective_closure, m.trainable_variables, options=dict(maxiter=100))

        print_summary(m)

        # mean and variance GP prediction

        y_pred, y_var = m.predict_f(X_test)
        y_pred = y_scaler.inverse_transform(y_pred)
        y_test = y_scaler.inverse_transform(y_test)
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
