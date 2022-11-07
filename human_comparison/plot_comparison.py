"""
Script for plotting the results of the MOGP model against human performance.
Author: Ryan-Rhys Griffiths
"""

from matplotlib import pyplot as plt

if __name__ == '__main__':

    mol_index = [1, 2, 3, 4, 5]
    mogp = [7.7, 6.1, 21.0, 21.2, 3.7]
    human = [18.4, 49.4, 101.1, 60.9, 84.5]
    best_human = [4, 57, 17, 0, 15]

    plt.plot(mol_index, mogp, marker='.', markersize=16, label='MOGP')
    plt.plot(mol_index, human, marker='.', markersize=16, label='Human')
    plt.plot(mol_index, best_human, marker='.', markersize=16, label='Best Human')
    plt.xticks([1, 2, 3, 4, 5], fontsize=12)
    plt.yticks([0, 20, 40, 60, 80, 100], fontsize=12)
    plt.xlabel('Molecule', fontsize=12)
    plt.ylabel('MAE (nm)', fontsize=12)
    corr1 = -0.1
    corr2 = 2.55
    corr3 = 0.1
    corr4 = -1
    for i, label in zip(mol_index, mogp):
        plt.annotate(str(label), xy=(i + corr1, label + corr2))
    for i, label in zip(mol_index, human):
        if i == 5:
            plt.annotate(str(label), xy=(i - 0.125, label + 2.1))
        else:
            plt.annotate(str(label), xy=(i + corr3, label + corr4))
    plt.legend(loc=2, fontsize=12)

    plt.xticks(ticks=[1, 2, 3, 4, 5], labels=['A', 'B', 'C', 'D', 'E'])
    plt.savefig('figures/human_performance_comparison_new.png')
