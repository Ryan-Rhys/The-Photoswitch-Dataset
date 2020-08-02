# Copyright Ryan-Rhys Griffiths and Aditya Raymond Thawani 2020
# Author: Ryan-Rhys Griffiths
"""
Script for performing inference on new candidates.
"""

import gpflow
from gpflow.mean_functions import Constant
from gpflow.utilities import print_summary
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler

from data_utils import TaskDataLoader, featurise_mols
from kernels import Tanimoto

representation = 'fingerprints'
task = 'e_iso_pi'
path = '../dataset/photoswitches.csv'

# New candidates to predict wavelength values for

candidate_list = ['O=C(OC)C(C=C1)=CC=C1C2=C[N-][N+]3=C(C=CN32)/N=N/C4=CC=CC=C4',
                  'O=C(OC)C(C=C1)=CC=C1C2=CN3[N+]([N-]2)=CC=C3/N=N/C4=CC=CC=C4']

if __name__ == '__main__':
    data_loader = TaskDataLoader(task, path)
    smiles_list, y_train = data_loader.load_property_data()
    X_train = featurise_mols(smiles_list, representation)
    X_test = featurise_mols(candidate_list, representation)

    num_features = np.shape(X_train)[1]

    # We define the Gaussian Process Regression Model using the Tanimoto kernel

    m = None

    def objective_closure():
        return -m.log_marginal_likelihood()

    #  We standardise the outputs but leave the inputs unchanged

    y_train = y_train.reshape(-1, 1)
    y_scaler = StandardScaler()
    y_train = y_scaler.fit_transform(y_train)

    X_train = X_train.astype(np.float64)
    X_test = X_test.astype(np.float64)

    # Fit GP

    k = Tanimoto()
    m = gpflow.models.GPR(data=(X_train, y_train), mean_function=Constant(np.mean(y_train)), kernel=k, noise_variance=1)

    # Optimise the kernel variance and noise level by the marginal likelihood

    opt = gpflow.optimizers.Scipy()
    opt.minimize(objective_closure, m.trainable_variables, options=dict(maxiter=100))
    print_summary(m)

    # mean and variance GP prediction

    y_pred, y_var = m.predict_f(X_test)
    y_pred = y_scaler.inverse_transform(y_pred)
    y_var = y_scaler.inverse_transform(y_var)

    print(f'GP {representation} prediction is ')
    print(y_pred)
    print(f'GP {representation} variance is')
    print(y_var)

    # Scale inputs for Random Forest

    x_scaler = StandardScaler()
    X_train_scaled = x_scaler.fit_transform(X_train)
    X_test_scaled = x_scaler.transform(X_test)

    regr_rf = RandomForestRegressor(n_estimators=1000, max_depth=300, random_state=2)
    regr_rf.fit(X_train, y_train)

    y_pred_rf = regr_rf.predict(X_test)
    y_pred_rf = y_scaler.inverse_transform(y_pred_rf)

    print(f'RF {representation} prediction is ')
    print(y_pred_rf)
