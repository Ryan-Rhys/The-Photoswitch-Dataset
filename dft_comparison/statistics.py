"""
Statistics for TD-DFT vs. MOGP comparison.
Author: Ryan-Rhys Griffiths
"""

import numpy as np
from matplotlib import pyplot as plt
import scipy
from scipy.stats import gaussian_kde, spearmanr

if __name__ == '__main__':

    # predictions

    ground_truth_vals_cam = np.loadtxt('dft_predictions/ground_truth_values_CAM-B3LYP.txt')
    ground_truth_vals_pbe = np.loadtxt('dft_predictions/ground_truth_values_PBE0.txt')
    mogp_pred_cam = np.loadtxt('dft_predictions/multioutput_gp_CAM-B3LYP.txt')
    mogp_pred_pbe = np.loadtxt('dft_predictions/multioutput_gp_PBE0.txt')
    tddft_pred_cam = np.loadtxt('dft_predictions/tddft_CAM-B3LYP.txt')
    tddft_pred_pbe = np.loadtxt('dft_predictions/tddft_PBE0.txt')
    tddft_pred_linear_cam = np.loadtxt('dft_predictions/tddft_linear_CAM-B3LYP.txt')
    tddft_pred_linear_pbe = np.loadtxt('dft_predictions/tddft_linear_PBE0.txt')

    pred_list_cam = [mogp_pred_cam, tddft_pred_cam, tddft_pred_linear_cam]
    pred_list_names_cam = ['MOGP CAM-B3LYP', 'TD-DFT CAM-B3LYP', 'TD-DFT CAM-B3LYP + Linear']
    pred_list_pbe = [mogp_pred_pbe, tddft_pred_pbe, tddft_pred_linear_pbe]
    pred_list_names_pbe = ['MOGP PBE0', 'TD-DFT PBE0', 'TD-DFT PBE0 + Linear']

    # errors

    mogp_errors_cam = np.loadtxt('dft_predictions/multioutput_gp_mse_CAM-B3LYP.txt')
    mogp_errors_pbeO = np.loadtxt('dft_predictions/multioutput_gp_mse_PBE0.txt')
    pbeO_errors = np.loadtxt('dft_predictions/tddft_mse_PBE0.txt')
    pbeO_errors = pbeO_errors[pbeO_errors > -100]
    cam_errors = np.loadtxt('dft_predictions/tddft_mse_CAM-B3LYP.txt')
    pbeO_linear_errors = np.loadtxt('dft_predictions/tddft_linear_mse_PBE0.txt')
    pbeO_linear_errors = pbeO_linear_errors[pbeO_linear_errors > -100]
    cam_linear_errors = np.loadtxt('dft_predictions/tddft_linear_mse_CAM-B3LYP.txt')

    error_list = [mogp_errors_cam, mogp_errors_pbeO, pbeO_errors, cam_errors, pbeO_linear_errors, cam_linear_errors]
    error_list_names = ['MOGP CAM-B3LYP', 'MOGP PBE0', 'TD-DFT CAM-B3LYP', 'TD-DFT PBE0', 'TD-DFT CAM-B3LYP + Linear', 'TD-DFT PBE0 + Linear']


    # plot the error distributions

    i = 0

    for error_dist in error_list:

        if i % 2 == 0:
            color = "#e65802"
        else:
            color = "#00b764"

        plt.hist(error_dist, bins=10, color=color, alpha=0.5, density=True)
        density = gaussian_kde(error_dist)
        xs = np.linspace(np.min(error_dist), np.max(error_dist), 200)
        plt.plot(xs, density(xs), color='k')
        plt.xticks(fontsize=12)
        # plt.yticks([0.1, 0.2, 0.3])
        plt.ylabel('Density', fontsize=16)
        plt.xlabel(f'Signed Error ({error_list_names[i]})', fontsize=16)
        plt.tight_layout()
        plt.savefig(f'figures/error_distribution_{error_list_names[i]}.png')
        plt.clf()

        i += 1

    i = 0

    for prediction in pred_list_cam:

        # Spearman rank-order correlation coefficient

        corr, _ = spearmanr(ground_truth_vals_cam, prediction)

        slope, intercept, r, p, stderr = scipy.stats.linregress(ground_truth_vals_cam, prediction)

        line = f'Regression line: y={intercept:.2f}+{slope:.2f}x, r={r:.2f}'

        fig, ax = plt.subplots()
        ax.plot(ground_truth_vals_cam, prediction, linewidth=0, marker='s', label=f'{pred_list_names_cam[i]}')
        ax.plot(ground_truth_vals_cam, intercept + slope * ground_truth_vals_cam, label=r"$\rho = $" + f'{corr:.2f}')
        ax.set_xlabel('Experimental Values (nm)')
        ax.set_ylabel(f' {pred_list_names_cam[i]} Prediction (nm)')
        ax.legend(facecolor='white')
        plt.savefig(f'figures/correlation_{pred_list_names_cam[i]}')

        i += 1

    i = 0

    for prediction in pred_list_pbe:

        corr, _ = spearmanr(ground_truth_vals_pbe, prediction)

        slope, intercept, r, p, stderr = scipy.stats.linregress(ground_truth_vals_pbe, prediction)

        line = f'Regression line: y={intercept:.2f}+{slope:.2f}x, r={r:.2f}'

        fig, ax = plt.subplots()
        ax.plot(ground_truth_vals_pbe, prediction, linewidth=0, marker='s', label=f'{pred_list_names_pbe[i]}')
        ax.plot(ground_truth_vals_pbe, intercept + slope * ground_truth_vals_pbe, label=r"$\rho = $" + f'{corr:.2f}')
        ax.set_xlabel('Experimental Values (nm)')
        ax.set_ylabel(f' {pred_list_names_pbe[i]} Prediction (nm)')
        ax.legend(facecolor='white')
        plt.savefig(f'figures/correlation_{pred_list_names_pbe[i]}')

        i += 1

