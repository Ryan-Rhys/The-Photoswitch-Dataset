# Copyright Ryan-Rhys Griffiths and Aditya Raymond Thawani 2020
# Author: Ryan-Rhys Griffiths
"""
Property prediction on the photoswitch dataset using Random Forest.
"""

import numpy as np
from rdkit.Chem import AllChem, Descriptors, MolFromSmiles
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error

from data_utils import load_e_iso_pi_data, load_thermal_data, load_z_iso_pi_data, load_e_iso_n_data, \
    load_z_iso_n_data, transform_data

PATH = '~/ml_physics/Photoswitches/dataset/photoswitches.csv'  # Change as appropriate
TASK = 'e_iso_pi'  # ['thermal', 'e_iso_pi', 'z_iso_pi', 'e_iso_n', 'z_iso_n']
use_fragments = False
use_pca = False


if __name__ == '__main__':

    if TASK == 'thermal':
        smiles_list, y = load_thermal_data(PATH)
    elif TASK == 'e_iso_pi':
        smiles_list, y = load_e_iso_pi_data(PATH)
    elif TASK == 'z_iso_pi':
        smiles_list, y = load_z_iso_pi_data(PATH)
    elif TASK == 'e_iso_n':
        smiles_list, y = load_e_iso_n_data(PATH)
    elif TASK == 'z_iso_n':
        smiles_list, y = load_z_iso_n_data(PATH)
    else:
        raise Exception('Must specify a valid task')

    if not use_fragments:

        feat = 'fingerprints'

        rdkit_mols = [MolFromSmiles(smiles) for smiles in smiles_list]
        X = [AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=512) for mol in rdkit_mols]
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

    if use_pca:
        n_components = 50
    else:
        n_components = None

    num_features = np.shape(X)[1]

    r2_list = []
    rmse_list = []
    mae_list = []

    print('\nBeginning training loop...')
    j = 0  # index for saving results

    for i in range(0, 25):

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=i)
        y_train = y_train.reshape(-1, 1)
        y_test = y_test.reshape(-1, 1)
        X_train, y_train, X_test, y_test, y_scaler = transform_data(X_train, y_train, X_test, y_test, n_components, use_pca)

        regr_rf = RandomForestRegressor(n_estimators=100, max_depth=30, random_state=2)
        regr_rf.fit(X_train, y_train)

        # Predict on new data
        y_rf = regr_rf.predict(X_test)
        y_rf = y_scaler.inverse_transform(y_rf)
        y_test = y_scaler.inverse_transform(y_test)
        score = r2_score(y_test, y_rf)
        rmse = np.sqrt(mean_squared_error(y_test, y_rf))
        mae = mean_absolute_error(y_test, y_rf)

        print("\nR^2: {:.3f}".format(score))
        print("RMSE: {:.3f}".format(rmse))
        print("MAE: {:.3f}".format(mae))

        r2_list.append(score)
        rmse_list.append(rmse)
        mae_list.append(mae)

        np.savetxt(TASK + '/results/random_forest/_seed_'+str(j)+'_ypred_'+feat+'.txt', y_rf)
        np.savetxt(TASK + '/results/random_forest/_seed_'+str(j)+'_ytest.txt', y_test)

        j += 1

    r2_list = np.array(r2_list)
    rmse_list = np.array(rmse_list)
    mae_list = np.array(mae_list)
    print("\nmean R^2: {:.4f} +- {:.4f}".format(np.mean(r2_list), np.std(r2_list)/np.sqrt(len(r2_list))))
    print("mean RMSE: {:.4f} +- {:.4f}".format(np.mean(rmse_list), np.std(rmse_list)/np.sqrt(len(rmse_list))))
    print("mean MAE: {:.4f} +- {:.4f}\n".format(np.mean(mae_list), np.std(mae_list)/np.sqrt(len(mae_list))))
