# Author: Arian Jamasb
"""
Property prediction using a Message-Passing Neural Network.
"""

import argparse

import dgl
import numpy as np
import torch
from dgllife.model.model_zoo import MPNNPredictor
from dgllife.utils import CanonicalAtomFeaturizer, CanonicalBondFeaturizer, mol_to_bigraph
from rdkit import Chem
from scipy.stats import pearsonr
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from torch.nn import MSELoss
from torch.utils.data import DataLoader

from data_utils import TaskDataLoader

if torch.cuda.is_available():
    print('use GPU')
    device = 'cuda'
else:
    print('use CPU')
    device = 'cpu'


def main(path, task, n_trials, test_set_size):
    """
    :param path: str specifying path to dataset.
    :param task: str specifying the task. One of ['e_iso_pi', 'z_iso_pi', 'e_iso_n', 'z_iso_n']
    :param n_trials: int specifying number of random train/test splits to use
    :param test_set_size: float in range [0, 1] specifying fraction of dataset to use as test set
    """

    data_loader = TaskDataLoader(task, path)
    smiles_list, y = data_loader.load_property_data()
    X = [Chem.MolFromSmiles(m) for m in smiles_list]

    # Collate Function for Dataloader
    def collate(sample):
        graphs, labels = map(list, zip(*sample))
        batched_graph = dgl.batch(graphs)
        batched_graph.set_n_initializer(dgl.init.zero_initializer)
        batched_graph.set_e_initializer(dgl.init.zero_initializer)
        return batched_graph, torch.tensor(labels)

    # Initialise featurisers
    atom_featurizer = CanonicalAtomFeaturizer()
    bond_featurizer = CanonicalBondFeaturizer()

    e_feats = bond_featurizer.feat_size('e')
    n_feats = atom_featurizer.feat_size('h')
    print('Number of features: ', n_feats)

    X = [mol_to_bigraph(m, node_featurizer=atom_featurizer, edge_featurizer=bond_featurizer) for m in X]

    r2_list = []
    rmse_list = []
    mae_list = []
    skipped_trials = 0

    for i in range(n_trials):

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_set_size, random_state=i+5)

        y_train = y_train.reshape(-1, 1)
        y_test = y_test.reshape(-1, 1)

        #  We standardise the outputs but leave the inputs unchanged

        y_scaler = StandardScaler()
        y_train_scaled = torch.Tensor(y_scaler.fit_transform(y_train))
        y_test_scaled = torch.Tensor(y_scaler.transform(y_test))

        train_data = list(zip(X_train, y_train_scaled))
        test_data = list(zip(X_test, y_test_scaled))

        train_loader = DataLoader(train_data, batch_size=32, shuffle=True, collate_fn=collate, drop_last=False)
        test_loader = DataLoader(test_data, batch_size=32, shuffle=False, collate_fn=collate, drop_last=False)

        mpnn_net = MPNNPredictor(node_in_feats=n_feats,
                                 edge_in_feats=e_feats
                                 )
        mpnn_net.to(device)

        loss_fn = MSELoss()
        optimizer = torch.optim.Adam(mpnn_net.parameters(), lr=0.001)

        mpnn_net.train()

        epoch_losses = []
        epoch_rmses = []
        for epoch in range(1, 201):
            epoch_loss = 0
            preds = []
            labs = []
            for i, (bg, labels) in enumerate(train_loader):
                labels = labels.to(device)
                atom_feats = bg.ndata.pop('h').to(device)
                bond_feats = bg.edata.pop('e').to(device)
                atom_feats, bond_feats, labels = atom_feats.to(device), bond_feats.to(device), labels.to(device)
                y_pred = mpnn_net(bg, atom_feats, bond_feats)
                labels = labels.unsqueeze(dim=1)
                loss = loss_fn(y_pred, labels)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                epoch_loss += loss.detach().item()

                # Inverse transform to get RMSE
                labels = y_scaler.inverse_transform(labels.reshape(-1, 1))
                y_pred = y_scaler.inverse_transform(y_pred.detach().numpy().reshape(-1, 1))

                # store labels and preds
                preds.append(y_pred)
                labs.append(labels)

            labs = np.concatenate(labs, axis=None)
            preds = np.concatenate(preds, axis=None)
            pearson, p = pearsonr(preds, labs)
            mae = mean_absolute_error(preds, labs)
            rmse = np.sqrt(mean_squared_error(preds, labs))
            r2 = r2_score(preds, labs)

            epoch_loss /= (i + 1)
            if epoch % 20 == 0:
                print(f"epoch: {epoch}, "
                      f"LOSS: {epoch_loss:.3f}, "
                      f"RMSE: {rmse:.3f}, "
                      f"MAE: {mae:.3f}, "
                      f"R: {pearson:.3f}, "
                      f"R2: {r2:.3f}")
            epoch_losses.append(epoch_loss)
            epoch_rmses.append(rmse)

        # Discount trial if train RMSE finishes as a negative value (optimiser error).

        if r2 < -1:
            skipped_trials += 1
            print('Skipped trials is {}'.format(skipped_trials))
            continue

        # Evaluate
        mpnn_net.eval()
        preds = []
        labs = []
        for i, (bg, labels) in enumerate(test_loader):
            labels = labels.to(device)
            atom_feats = bg.ndata.pop('h').to(device)
            bond_feats = bg.edata.pop('e').to(device)
            atom_feats, bond_feats, labels = atom_feats.to(device), bond_feats.to(device), labels.to(device)
            y_pred = mpnn_net(bg, atom_feats, bond_feats)
            labels = labels.unsqueeze(dim=1)

            # Inverse transform to get RMSE
            labels = y_scaler.inverse_transform(labels.reshape(-1, 1))
            y_pred = y_scaler.inverse_transform(y_pred.detach().numpy().reshape(-1, 1))

            preds.append(y_pred)
            labs.append(labels)

        labs = np.concatenate(labs, axis=None)
        preds = np.concatenate(preds, axis=None)

        pearson, p = pearsonr(preds, labs)
        mae = mean_absolute_error(preds, labs)
        rmse = np.sqrt(mean_squared_error(preds, labs))
        r2 = r2_score(preds, labs)

        r2_list.append(r2)
        rmse_list.append(rmse)
        mae_list.append(mae)

        print(f'Test RMSE: {rmse:.3f}, MAE: {mae:.3f}, R: {pearson:.3f}, R2: {r2:.3f}')

    r2_list = np.array(r2_list)
    rmse_list = np.array(rmse_list)
    mae_list = np.array(mae_list)

    print("\nmean R^2: {:.4f} +- {:.4f}".format(np.mean(r2_list), np.std(r2_list)/np.sqrt(len(r2_list))))
    print("mean RMSE: {:.4f} +- {:.4f}".format(np.mean(rmse_list), np.std(rmse_list)/np.sqrt(len(rmse_list))))
    print("mean MAE: {:.4f} +- {:.4f}\n".format(np.mean(mae_list), np.std(mae_list)/np.sqrt(len(mae_list))))
    print("\nSkipped trials is {}".format(skipped_trials))


if __name__ == '__main__':

    parser = argparse.ArgumentParser()

    parser.add_argument('-p', '--path', type=str, default='../dataset/photoswitches.csv',
                        help='Path to the photoswitches.csv file.')
    parser.add_argument('-t', '--task', type=str, default='e_iso_pi',
                        help='str specifying the task. One of [e_iso_pi, z_iso_pi, e_iso_n, z_iso_n].')
    parser.add_argument('-n', '--n_trials', type=int, default=20,
                        help='int specifying number of random train/test splits to use')
    parser.add_argument('-ts', '--test_set_size', type=float, default=0.2,
                        help='float in range [0, 1] specifying fraction of dataset to use as test set')

    args = parser.parse_args()

    main(args.path, args.task, args.n_trials, args.test_set_size)
