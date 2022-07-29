"""
Comparing the novelty i.e. similarity of the discovered candidates against the molecules from the training set.
Author: Ryan-Rhys Griffiths
"""

from data_utils import TaskDataLoader, featurise_mols
from kernels import Tanimoto

import numpy as np


def tan_similarity(X, Y):
    "Tanimoto similarity function"

    X1s = np.sum(np.square(X))  # Squared L2-norm of X
    X2s = np.sum(np.square(Y))  # Squared L2-norm of Y
    inner_product = np.dot(X, Y)  # inner product of the X and Y
    denominator = -inner_product + (X1s + X2s)
    out = inner_product / denominator

    return out


if __name__ == '__main__':

    path = '../dataset/photoswitches.csv'
    task = 'e_iso_pi'

    candidate_list = ['CCOC1=CC=C(/N=N/C2=C3N=CC(C#N)=C(N)N3N=C2N)C=C1',
                      'NC1=NN2C(N)=C(C#N)C=NC2=C1/N=N/C3=CC4=C(OCO4)C=C3',
                      'COC1=CC=C(C2=NC3=C(C(N)=NN3C(N)=C2C#N)/N=N/C4=CC5=C(OCO5)C=C4)C=C1',
                      'O=S(C1=CC=C(/N=N/C2=C3N=C(C4=CC=CS4)C(C#N)=C(N)N3N=C2N)C=C1)(N)=O',
                      'CSC1=C(C(N)=NC2=C(C(N)=NN12)/N=N/C3=CC=C(S(N)(=O)=O)C=C3)C#N',
                      'COC1=CC=C(/N=N/C2=C3N=C(C4=CC=CC=C4)C(C#N)=C(N)N3N=C2N)C=C1',
                      '[O-]Cl(=O)(=O)=O.CCN(C1=CC=C(/N=N/C2=C([N+]([O-])=O)C(C)=[N+](N2C)C)C=C1)CC',
                      'CN(C1=CC=C(/N=N/C2=NC(C#N)=C(C#N)N2)C=C1)C',
                      'CN1C(/N=N/C2=CC=C(NC3=CC=CC=C3)C=C2)=NC4=CC=CC=C14',
                      'CCN(S(=O)(C1=CC=C(/N=N/C2=C(C3=CC=CC=C3)N=C(N)S2)C=C1)=O)CC',
                      'O=C1N(C2=CC=CC=C2)N(C)C(C)=C1/N=N/C3=CC=C(N)C(OC)=C3']

    # Satisfying both criteria
    good_candidates = [candidate_list[4], candidate_list[5], candidate_list[8], candidate_list[9], candidate_list[10], candidate_list[7]]
    candidate_fingerprints = featurise_mols(good_candidates, 'fingerprints')

    data_loader = TaskDataLoader(task, path)
    smiles_list, y = data_loader.load_property_data()
    X = featurise_mols(smiles_list, 'fingerprints')

    mol_id = 0
    for candidate in candidate_fingerprints:
        candidate_similarities = []
        i = 0
        while i < X.shape[0]:
            sim = tan_similarity(X[i], candidate)
            candidate_similarities.append(sim)
            i += 1
        ranked_sim_list = np.argsort(candidate_similarities, axis=0).flatten()
        ranked_smiles = np.array(smiles_list)[ranked_sim_list][-3:]
        if mol_id == 0:
            id = 5
        elif mol_id == 1:
            id = 6
        elif mol_id == 2:
            id = 9
        elif mol_id == 3:
            id = 10
        elif mol_id == 4:
            id = 11
        elif mol_id == 5:
            id = 8
        with open(f'novelty_analysis/closest_3_smiles_molecule_{id}.txt', 'w') as f:
            f.write(f'Discovered Candidate SMILES is: {good_candidates[mol_id]}' + '\n')
            f.write(f'Closest in training set are:' + '\n')
            for smiles in ranked_smiles:
                f.write(smiles + '\n')
            f.close()
        mol_id += 1
