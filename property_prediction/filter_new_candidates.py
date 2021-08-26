# Copyright Ryan-Rhys Griffiths and Aditya Raymond Thawani 2020
# Author: Ryan-Rhys Griffiths
"""
Script for filtering purchasable candidates. Criterion for new candidates is that the E isomer pi-pi*
value be between 450-600nm. The separation between the E isomer n-pi* and Z isomer n-pi* is not less than 15nm.
The separation between E isomer pi-pi* and Z isomer pi-pi* is greater than 40nm.
"""

import numpy as np

if __name__ == '__main__':

    # Load property predictions for the purchasable molecules for a given model.

    model = 'multioutput_gp'  # ['gp', 'rf', 'ensemble', 'multioutput_gp']

    data_ep_mgp = np.loadtxt(f'../property_prediction/predictions/purchasable_{model}_task_e_iso_pi.txt')
    data_en_mgp = np.loadtxt(f'../property_prediction/predictions/purchasable_{model}_task_e_iso_n.txt')
    data_zn_mgp = np.loadtxt(f'../property_prediction/predictions/purchasable_{model}_task_z_iso_n.txt')
    data_zp_mgp = np.loadtxt(f'../property_prediction/predictions/purchasable_{model}_task_z_iso_pi.txt')

    model = 'gp'  # ['gp', 'rf', 'ensemble', 'multioutput_gp']

    data_ep_gp = np.loadtxt(f'../property_prediction/predictions/purchasable_{model}_task_e_iso_pi.txt')
    data_en_gp = np.loadtxt(f'../property_prediction/predictions/purchasable_{model}_task_e_iso_n.txt')
    data_zn_gp = np.loadtxt(f'../property_prediction/predictions/purchasable_{model}_task_z_iso_n.txt')
    data_zp_gp = np.loadtxt(f'../property_prediction/predictions/purchasable_{model}_task_z_iso_pi.txt')

    model = 'rf'  # ['gp', 'rf', 'ensemble', 'multioutput_gp']

    data_ep_rf = np.loadtxt(f'../property_prediction/predictions/purchasable_{model}_task_e_iso_pi.txt')
    data_en_rf = np.loadtxt(f'../property_prediction/predictions/purchasable_{model}_task_e_iso_n.txt')
    data_zn_rf = np.loadtxt(f'../property_prediction/predictions/purchasable_{model}_task_z_iso_n.txt')
    data_zp_rf = np.loadtxt(f'../property_prediction/predictions/purchasable_{model}_task_z_iso_pi.txt')

    data_ep = (data_ep_gp + data_ep_rf + data_ep_mgp)/3.0
    data_en = (data_en_gp + data_en_rf + data_en_mgp)/3.0
    data_zp = (data_zp_gp + data_zp_rf + data_zp_mgp)/3.0
    data_zn = (data_zn_gp + data_zn_rf + data_zn_mgp)/3.0

    # Apply the first criterion

    model = 'triple_ensemble'

    data_filtered_crit_one = np.where(data_ep > 450)
    data_filtered_crit_two = np.abs(data_en - data_zn)
    data_filtered_crit_two = np.where(data_filtered_crit_two < 15)

    # Collect the indices of molecules that satisfy criteria 1 and 2

    satisfy_two = np.intersect1d(data_filtered_crit_one, data_filtered_crit_two)

    data_filtered_crit_three = np.abs(data_ep - data_zp)
    data_filtered_crit_three = np.where(data_filtered_crit_three > 40)

    # Collect the indices of molecules that satisfy criteria 1, 2 and 3.

    satisfy_three = np.intersect1d(satisfy_two, data_filtered_crit_three)

    with open(f'predictions/predicted_smiles/{model}.txt', 'w') as file_to_write:
        with open('../dataset/purchasable_switch.csv', 'r') as file_to_read:
            i = -1
            for line in file_to_read:
                if i in satisfy_three:
                    file_to_write.write(line)
                i += 1
