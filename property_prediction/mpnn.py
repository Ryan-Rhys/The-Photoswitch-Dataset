import os
from rdkit import Chem
from rdkit import RDPaths
import numpy as np
import pandas as pd
import torch
import dgl
if torch.cuda.is_available():
    print('use GPU')
    device = 'cuda'
else:
    print('use CPU')
    device = 'cpu'
from dgllife.utils import CanonicalAtomFeaturizer, CanonicalBondFeaturizer, mol_to_complete_graph, mol_to_graph, mol_to_bigraph
from dgllife.model.model_zoo import MPNNPredictor
from torch.utils.data import DataLoader
from torch.nn import MSELoss
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

TASK_NAME = ''
DATA_PATH = ''


if __name__ == '__main__':

    # Load data
    df = pd.read_csv('../dataset/photoswitches.csv')
    df = df.loc[~df['E isomer pi-pi* wavelength in nm'].isna()]
    #print(df)

    scaler = MinMaxScaler()
    scaler.fit(np.asarray(df['E isomer pi-pi* wavelength in nm']).reshape(-1, 1))

    # Create Train-Test Splits
    X_train, X_test = train_test_split(df, test_size=0.2)

    # Loads Mols from SMILES
    trainmols = [Chem.MolFromSmiles(m) for m in X_train['SMILES']]
    testmols = [Chem.MolFromSmiles(m) for m in X_test['SMILES']]

    # Initialse featurisers
    atom_featurizer = CanonicalAtomFeaturizer()
    bond_featurizer = CanonicalBondFeaturizer()
    n_feats = atom_featurizer.feat_size('h')
    e_feats = bond_featurizer.feat_size('e')
    print('Number of features: ', n_feats)

    # Create graphs and labels
    train_g = [mol_to_bigraph(m, node_featurizer=atom_featurizer, edge_featurizer=bond_featurizer) for m in trainmols]
    train_y = torch.Tensor(scaler.transform(np.asarray(X_train['E isomer pi-pi* wavelength in nm']).reshape(-1,1)))

    test_g = [mol_to_bigraph(m, node_featurizer=atom_featurizer, edge_featurizer=bond_featurizer) for m in testmols]
    test_y = torch.Tensor(scaler.transform(np.asarray(X_train['Z isomer pi-pi* wavelength in nm']).reshape(-1, 1)))

    # Collate Function for Dataloader
    def collate(sample):
        graphs, labels = map(list, zip(*sample))
        batched_graph = dgl.batch(graphs)
        batched_graph.set_n_initializer(dgl.init.zero_initializer)
        batched_graph.set_e_initializer(dgl.init.zero_initializer)
        return batched_graph, torch.tensor(labels)

    train_data = list(zip(train_g, train_y))
    test_data = list(zip(test_g, test_y))

    train_loader = DataLoader(train_data, batch_size=32, shuffle=True, collate_fn=collate, drop_last=True)
    test_loader = DataLoader(test_data, batch_size=32, shuffle=False, collate_fn=collate, drop_last=True)

    #gcn_net = GCNPredictor(in_feats=n_feats,
    #                       hidden_feats=[64,32],
    #                       #activation=['relu', 'relu'],
    #                       batchnorm=[True, False],
    #                       dropout=[0.3,0],
    #                       classifier_hidden_feats=1
    #                       )

    mpnn_net = MPNNPredictor(node_in_feats=n_feats,
                             #node_hidden_feats=[64.32],
                             #batch_norm=[True, False],
                             #dropout=[0.3,0],
                             #classifier_hidden_feats=1,
                             edge_in_feats=e_feats
                             )

    mpnn_net.to(device)

    loss_fn = MSELoss()
    optimizer = torch.optim.Adam(mpnn_net.parameters(), lr=0.001)

    mpnn_net.train()

    epoch_losses = []
    epoch_rmses = []
    #epoch_accuracies = []
    for epoch in range(1,1001):
        epoch_loss = 0
        epoch_rmse = 0
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

            #Inverse transform to get RMSE
            labels = scaler.inverse_transform(labels.reshape(-1, 1))
            y_pred = scaler.inverse_transform(y_pred.detach().numpy().reshape(-1, 1))
            rmse = np.divide(np.sum(np.sqrt(np.square(y_pred - labels))), 32)
            epoch_rmse += rmse

        epoch_rmse /= (i +1)
        epoch_loss /= (i + 1)
        if epoch % 20 == 0:
            print(f"epoch: {epoch}, LOSS: {epoch_loss:.3f}, RMSE: {epoch_rmse:.3f}")
        epoch_losses.append(epoch_loss)
        epoch_rmses.append(epoch_rmse)

    # Evaluate
    mpnn_net.eval()
    test_loss = 0
    squared_errors = []
    for i, (bg, labels) in enumerate(test_loader):
        labels = labels.to(device)
        atom_feats = bg.ndata.pop('h').to(device)
        bond_feats = bg.edata.pop('e').to(device)
        atom_feats, bond_feats, labels = atom_feats.to(device), bond_feats.to(device), labels.to(device)
        y_pred = mpnn_net(bg, atom_feats, bond_feats)
        labels = labels.unsqueeze(dim=1)

        #Inverse transform to get RMSE
        labels = scaler.inverse_transform(labels.reshape(-1, 1))
        y_pred = scaler.inverse_transform(y_pred.detach().numpy().reshape(-1, 1))
        se = float(np.sum(np.sqrt(np.square(y_pred - labels))))
        squared_errors.append(se)
    print(squared_errors)
    print(test_y.shape)





