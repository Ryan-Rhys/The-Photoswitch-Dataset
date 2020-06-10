# Copyright Ryan-Rhys Griffiths and Aditya Raymond Thawani 2020
# Author: Ryan-Rhys Griffiths
"""
Script to perform data vizualisation
"""

from matplotlib import pyplot as plt
import numpy as np
from rdkit.Chem import Descriptors, MolFromSmiles
from rdkit.Chem.AllChem import GetMorganFingerprintAsBitVect
from sklearn.decomposition import PCA, KernelPCA
from sklearn.preprocessing import StandardScaler
import umap
import seaborn as sns

from property_prediction.data_utils import TaskDataLoader


PATH = '~/ml_physics/Photoswitches/dataset/data_viz.csv'
use_fragments = False


if __name__ == '__main__':

    data_loader = TaskDataLoader(task='thermal', path=PATH)
    smiles_list, _ = data_loader.load_property_data()

    if not use_fragments:

        feat = 'fingerprints'

        rdkit_mols = [MolFromSmiles(smiles) for smiles in smiles_list]
        X = [GetMorganFingerprintAsBitVect(mol, 2, nBits=512) for mol in rdkit_mols]
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

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    pca = PCA(n_components=2)
    embedding_pca = pca.fit_transform(X_scaled)
    print('Fraction of variance retained is: ' + str(sum(pca.explained_variance_ratio_)))

    plt.scatter(embedding_pca[:, 0], embedding_pca[:, 1])
    plt.gca().set_aspect('equal', 'datalim')
    plt.title('PCA projection of the Photoswitch dataset', fontsize=12)
    plt.savefig('visualization_figures/PCA_projection_use_frags_is_{}'.format(use_fragments))
    plt.show()

    kpca = KernelPCA(n_components=2, kernel='cosine')
    embedding_kpca = kpca.fit_transform(X_scaled)

    plt.scatter(embedding_kpca[:, 0], embedding_kpca[:, 1])
    plt.gca().set_aspect('equal', 'datalim')
    plt.title('KPCA projection of the Photoswitch dataset', fontsize=12)
    plt.savefig('visualization_figures/KPCA_projection_use_frags_is_{}'.format(use_fragments))
    plt.show()

    reducer = umap.UMAP(n_neighbors=50, min_dist=0.01)
    embedding_umap = reducer.fit_transform(X, )
    print(embedding_umap.shape)

    sns.set(style='white', context='notebook', rc={'figure.figsize': (14, 10)})

    plt.scatter(embedding_umap[:, 0], embedding_umap[:, 1])
    plt.gca().set_aspect('equal', 'datalim')
    plt.title('UMAP projection of the Photoswitch dataset', fontsize=12)
    plt.savefig('visualization_figures/UMAP_projection_use_frags_is_{}'.format(use_fragments))
    plt.show()
