# Author: Ryan-Rhys Griffiths
"""
Hyperparameter tuning for the Graph Attention Network.
"""

import argparse

import dgl
import numpy as np
import torch
from dgllife.model.model_zoo import GATPredictor
from dgllife.utils import CanonicalAtomFeaturizer, mol_to_bigraph
from rdkit import Chem
from scipy.stats import pearsonr
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from sklearn.model_selection import train_test_split, KFold
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


def main(path, task):
    """
    :param path: str specifying path to dataset.
    :param task: str specifying the task. One of ['e_iso_pi', 'z_iso_pi', 'e_iso_n', 'z_iso_n']
    """

    data_loader = TaskDataLoader(task, path)
    X, y = data_loader.load_property_data()
    X = [Chem.MolFromSmiles(m) for m in X]

    # Collate Function for Dataloader
    def collate(sample):
        graphs, labels = map(list, zip(*sample))
        batched_graph = dgl.batch(graphs)
        batched_graph.set_n_initializer(dgl.init.zero_initializer)
        batched_graph.set_e_initializer(dgl.init.zero_initializer)
        return batched_graph, torch.tensor(labels)

    # Initialise featurisers
    atom_featurizer = CanonicalAtomFeaturizer()
    n_feats = atom_featurizer.feat_size('h')
    print('Number of features: ', n_feats)

    X_full, _, y_full, _ = train_test_split(X, y, test_size=0.2, random_state=30)
    y_full = y_full.reshape(-1, 1)

    #  We standardise the outputs but leave the inputs unchanged

    y_scaler = StandardScaler()
    y_full = torch.Tensor(y_scaler.fit_transform(y_full))

    # Set up cross-validation splits

    n_splits = 5
    kf = KFold(n_splits=n_splits)

    X_train_splits = []
    y_train_splits = []
    X_val_splits = []
    y_val_splits = []

    for train_index, test_index in kf.split(X_full):
        X_train, X_val = np.array(X_full)[train_index], np.array(X_full)[test_index]
        y_train, y_val = y_full[train_index], y_full[test_index]
        # Create graphs and labels
        X_train = [mol_to_bigraph(m, node_featurizer=atom_featurizer) for m in X]
        X_val = [mol_to_bigraph(m, node_featurizer=atom_featurizer) for m in X_val]
        X_train_splits.append(X_train)
        X_val_splits.append(X_val)
        y_train_splits.append(y_train)
        y_val_splits.append(y_val)

    def lognuniform(low=1, high=5, size=None, base=10):
        return np.power(base, -np.random.uniform(low, high, size))

    best_rmse = 100000000

    for i in range(1000):

        num_layers = np.random.randint(1, 4)
        classifier_hidden_feats = np.random.randint(1, 128)
        hidden_feats = [np.random.choice([16, 32, 64])]*num_layers
        num_heads = [np.random.randint(2, 16)]*num_layers
        attn_drops = [np.random.uniform(0, 0.5)]*num_layers
        feat_drops = [np.random.uniform(0, 0.5)]*num_layers
        learning_rate = lognuniform()

        param_set = {'num_layers': num_layers, 'classifier_hidden_feats': classifier_hidden_feats, 'num_heads': num_heads,
                     'hidden_feats': hidden_feats, 'dropout': feat_drops, 'attn_dropout': attn_drops, 'lr': learning_rate}

        print(f'\nParameter set in trial {i} is \n')
        print(param_set)
        print('\n')

        cv_rmse_list = []

        for j in range(n_splits):

            X_train = X_train_splits[j]
            y_train = y_train_splits[j]
            X_val = X_val_splits[j]
            y_val = y_val_splits[j]

            train_data = list(zip(X_train, y_train))
            test_data = list(zip(X_val, y_val))

            train_loader = DataLoader(train_data, batch_size=32, shuffle=True, collate_fn=collate, drop_last=False)
            test_loader = DataLoader(test_data, batch_size=32, shuffle=False, collate_fn=collate, drop_last=False)

            gat_net = GATPredictor(in_feats=n_feats,
                                   hidden_feats=hidden_feats,
                                   num_heads=num_heads,
                                   feat_drops=feat_drops,
                                   attn_drops=attn_drops,
                                   classifier_hidden_feats=classifier_hidden_feats)

            gat_net.to(device)

            loss_fn = MSELoss()
            optimizer = torch.optim.Adam(gat_net.parameters(), lr=learning_rate)

            gat_net.train()

            epoch_losses = []
            epoch_rmses = []
            for epoch in range(1, 501):
                epoch_loss = 0
                preds = []
                labs = []
                for i, (bg, labels) in enumerate(train_loader):
                    labels = labels.to(device)
                    atom_feats = bg.ndata.pop('h').to(device)
                    atom_feats, labels = atom_feats.to(device), labels.to(device)
                    y_pred = gat_net(bg, atom_feats)
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

            # Evaluate
            gat_net.eval()
            preds = []
            labs = []
            for i, (bg, labels) in enumerate(test_loader):
                labels = labels.to(device)
                atom_feats = bg.ndata.pop('h').to(device)
                atom_feats, labels = atom_feats.to(device), labels.to(device)
                y_pred = gat_net(bg, atom_feats)
                labels = labels.unsqueeze(dim=1)

                # Inverse transform to get RMSE
                labels = y_scaler.inverse_transform(labels.reshape(-1, 1))
                y_pred = y_scaler.inverse_transform(y_pred.detach().numpy().reshape(-1, 1))

                preds.append(y_pred)
                labs.append(labels)

            preds = np.concatenate(preds, axis=None)
            labs = np.concatenate(labs, axis=None)

            pearson, p = pearsonr(preds, labs)
            mae = mean_absolute_error(preds, labs)
            rmse = np.sqrt(mean_squared_error(preds, labs))
            cv_rmse_list.append(rmse)
            r2 = r2_score(preds, labs)

            print(f'Test RMSE: {rmse:.3f}, MAE: {mae:.3f}, R: {pearson:.3f}, R2: {r2:.3f}')

        param_rmse = np.mean(cv_rmse_list)
        if param_rmse < best_rmse:
            best_rmse = param_rmse
            best_params = param_set

    print('Best RMSE and best params \n')
    print(best_rmse)
    print(best_params)
    np.savetxt('saved_hypers/GAT', best_params)


if __name__ == '__main__':

    parser = argparse.ArgumentParser()

    parser.add_argument('-p', '--path', type=str, default='../dataset/photoswitches.csv',
                        help='Path to the photoswitches.csv file.')
    parser.add_argument('-t', '--task', type=str, default='e_iso_pi',
                        help='str specifying the task. One of [e_iso_pi, z_iso_pi, e_iso_n, z_iso_n].')

    args = parser.parse_args()

    main(args.path, args.task)
