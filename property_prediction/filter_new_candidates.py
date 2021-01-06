# Copyright Ryan-Rhys Griffiths and Aditya Raymond Thawani 2020
# Author: Ryan-Rhys Griffiths
"""
Script for filtering purchasable candidates. Criterion for new candidates is that the E isomer pi-pi*
value be between 450-600nm. The separation between the E isomer n-pi* and Z isomer n-pi* is not less than 15nm.
The separation between E isomer pi-pi* and Z isomer pi-pi* is greater than 40nm.
"""

import numpy as np

model = 'gp'  # ['gp', 'rf', 'ensemble', 'multioutput_gp']

if __name__ == '__main__':

    # Load property predictions for the purchasble molecules for a given model.

    data_ep = np.loadtxt(f'../property_prediction/predictions/purchasable_{model}_task_e_iso_pi.txt')
    data_en = np.loadtxt(f'../property_prediction/predictions/purchasable_{model}_task_e_iso_n.txt')
    data_zn = np.loadtxt(f'../property_prediction/predictions/purchasable_{model}_task_z_iso_n.txt')
    data_zp = np.loadtxt(f'../property_prediction/predictions/purchasable_{model}_task_z_iso_pi.txt')

    # Apply the first criterion

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
