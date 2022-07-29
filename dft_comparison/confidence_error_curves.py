"""
Code for producing confidence-error curves of the MOGP predictions.
Author: Ryan-Rhys Griffiths
"""

import matplotlib.pyplot as plt
import numpy as np

if __name__ == '__main__':

    theory_level = 'CAM-B3LYP' # One of ['CAM-B3LYP', 'PBE0']
    if theory_level == 'CAM-B3LYP':
        size = 99
    else:  # PBE0
        size = 114

    # load the MOGP variance for each heldout point
    y_var = np.loadtxt(f'dft_predictions/var_values_{theory_level}.txt')
    y_test = np.loadtxt(f'dft_predictions/ground_truth_values_{theory_level}.txt')
    y_pred = np.loadtxt(f'dft_predictions/multioutput_gp_{theory_level}.txt')

    mae_confidence_list = []

    # rank the points by variance
    ranked_confidence_list = np.argsort(y_var, axis=0).flatten()

    for k in range(size):
        conf = ranked_confidence_list[0:k + 1]

        # for cumulative plots
        # mae = np.mean(np.abs(y_test[conf] - y_pred[conf]))

        # Construct the MAE error for each level of confidence
        mae = np.abs(y_test[ranked_confidence_list[k]] - y_pred[ranked_confidence_list[k]])
        mae_confidence_list.append(mae)

    confidence_percentiles = np.arange(1e-14, 100, 100/size)  # 1e-14 instead of 0 to stop weirdness with len(y_test) = 29
    mae_confidence = np.array(mae_confidence_list)
    mae_confidence = np.flip(mae_confidence)  # flip so that 100th percentile is most confident

    window = 50
    average_data = []
    for ind in range(len(mae_confidence) - window + 1):
        average_data.append(np.mean(mae_confidence[ind:ind + window]))

    for ind in range(window - 1):
        average_data.insert(0, np.nan)

    print(np.mean(mae_confidence[0:int(size/2)]))
    print(np.mean(mae_confidence[int(size/2):]))

    # We plot the Mean-absolute error confidence-error curve

    plt.plot(confidence_percentiles, mae_confidence, label='MAE')
    plt.plot(confidence_percentiles, average_data, label='Moving Average')
    plt.xlabel('Confidence Percentile')
    plt.ylabel('MAE (nm)')
    plt.ylim([0, np.max(mae_confidence) + 1])
    plt.xlim([0, 100 * ((size - 1) / size)])
    plt.yticks(np.arange(0, np.max(mae_confidence) + 1, 10.0))
    plt.legend(loc=1)
    plt.savefig(f'confidence_error_curves/mae_{theory_level}.png')
    plt.show()
