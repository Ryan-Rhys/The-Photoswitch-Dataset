"""
Script for computing the Wilcoxon signed-rank test on the E isomer pi-pi* benchmark. chosen based on:
https://machinelearningmastery.com/statistical-significance-tests-for-comparing-machine-learning-algorithms/
and https://ieeexplore.ieee.org/document/6790639 (Dietterich 1998).
"""

from scipy.stats import wilcoxon

if __name__ == '__main__':

    # E isomer pi-pi*

    gp_maes = [11.243, 12.975, 13.785, 10.551, 14.573, 14.910, 12.309, 14.756, 14.549, 11.172, 13.898, 11.560, 15.228, 11.293,
               13.063, 12.250, 13.920, 15.103, 13.896, 14.584]
    mogp_maes = [10.872, 13.290, 14.101, 10.829, 13.894, 15.081, 11.946, 15.155, 14.667, 10.786, 13.889, 11.860, 15.116,
                 11.185, 13.263, 12.105, 12.221, 15.043, 13.181, 12.788]

    print(wilcoxon(gp_maes, mogp_maes))

    # Z isomer pi-pi*

    gp_maes_z = [9.214, 9.629, 10.121, 8.686, 8.721, 12.056, 8.982, 9.865, 8.842, 8.520, 8.144, 9.281, 11.103, 8.590,
                 10.883, 8.056, 11.732, 10.437, 15.543, 7.208]
    mogp_maes_z = [8.820, 7.539, 9.532, 8.934, 6.238, 9.978, 8.044, 8.344, 10.478, 7.570, 7.399, 8.469, 7.550, 9.592,
                   12.450, 9.996, 7.812, 7.570, 11.425, 9.008]

    print(wilcoxon(gp_maes_z, mogp_maes_z))
