# Copyright Ryan-Rhys Griffiths and Aditya Raymond Thawani 2020
# Author: Ryan-Rhys Griffiths
"""
Script for performing inference on new candidates. Criterion for new candidates is that the E isomer pi-pi*
value be between 450-600nm. The separation between the E isomer n-pi* and Z isomer n-pi* is not less than 15nm.
The separation between E isomer pi-pi* and Z isomer pi-pi* is greater than 40nm.
"""

import numpy as np
import pandas as pd

if __name__ == '__main__':

    # Criterion 1

    data_gp_ep = np.loadtxt('../property_prediction/predictions/purchasable_gp_task_e_iso_pi.txt')
    data_filtered_gp = np.where(data_gp_ep > 450)

    # Criterion 2

    data_gp_en = np.loadtxt('../property_prediction/predictions/purchasable_gp_task_e_iso_n.txt')
    data_gp_zn = np.loadtxt('../property_prediction/predictions/purchasable_gp_task_z_iso_n.txt')
    data_crit_two = np.abs(data_gp_en - data_gp_zn)
    data_filtered_crit_two = np.where(data_crit_two < 15)

    satisfy_two = np.intersect1d(data_filtered_gp, data_filtered_crit_two)

    # Criterion 3

    data_gp_zp = np.loadtxt('../property_prediction/predictions/purchasable_gp_task_z_iso_pi.txt')
    data_crit_three = np.abs(data_gp_ep - data_gp_zp)
    data_filtered_crit_three = np.where(data_crit_three > 40)

    satisfy_three_gp = np.intersect1d(satisfy_two, data_filtered_crit_three)

    # Random Forest

    # Criterion 1

    data_rf_ep = np.loadtxt('../property_prediction/predictions/purchasable_rf_task_e_iso_pi.txt')
    data_filtered_rf = np.where(data_rf_ep > 450)

    # Criterion 2

    data_rf_en = np.loadtxt('../property_prediction/predictions/purchasable_rf_task_e_iso_n.txt')
    data_rf_zn = np.loadtxt('../property_prediction/predictions/purchasable_rf_task_z_iso_n.txt')
    data_crit_two = np.abs(data_rf_en - data_rf_zn)
    data_filtered_crit_two = np.where(data_crit_two < 15)

    satisfy_two = np.intersect1d(data_filtered_rf, data_filtered_crit_two)

    # Criterion 3

    data_rf_zp = np.loadtxt('../property_prediction/predictions/purchasable_rf_task_z_iso_pi.txt')
    data_crit_three = np.abs(data_rf_ep - data_rf_zp)
    data_filtered_crit_three = np.where(data_crit_three > 40)

    satisfy_three = np.intersect1d(satisfy_two, data_filtered_crit_three)

    # Ensemble

    print('hi')

    # Criterion 1

    data_ensemble_ep = np.loadtxt('../property_prediction/predictions/purchasable_ensemble_task_e_iso_pi.txt')
    data_filtered_ensemble = np.where(data_ensemble_ep > 450)

    print('hi')

    # Criterion 2

    data_ensemble_en = np.loadtxt('../property_prediction/predictions/purchasable_ensemble_task_e_iso_n.txt')
    data_ensemble_zn = np.loadtxt('../property_prediction/predictions/purchasable_ensemble_task_z_iso_n.txt')
    data_crit_two = np.abs(data_ensemble_en - data_ensemble_zn)
    data_filtered_crit_two = np.where(data_crit_two < 15)

    print('hi')

    satisfy_two = np.intersect1d(data_filtered_ensemble, data_filtered_crit_two)

    # Criterion 3

    data_ensemble_zp = np.loadtxt('../property_prediction/predictions/purchasable_ensemble_task_z_iso_pi.txt')
    data_crit_three = np.abs(data_ensemble_ep - data_ensemble_zp)
    data_filtered_crit_three = np.where(data_crit_three > 40)

    satisfy_three_ensemble = np.intersect1d(satisfy_two, data_filtered_crit_three)

    print('hi')
