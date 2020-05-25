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
from dgllife.model.model_zoo import GCNPredictor
from torch.utils.data import DataLoader
from torch.nn import MSELoss
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from scipy.stats import pearsonr
from sklearn.metrics import r2_score

TASK_NAME = 'e_iso_pi'  # ['thermal', 'e_iso_pi', 'z_iso_pi', 'e_iso_n', 'z_iso_n']
DATA_PATH = '../dataset/photoswitches.csv'
GRAPH_TYPE = 'bigraph' #['bigraph', 'complete']

if __name__ == '__main__':

    if TASK_NAME == 'e_iso_pi':
        label_col = 'E isomer pi-pi* wavelength in nm'
    elif TASK_NAME == 'z_iso_pi':
        label_col = 'Z isomer pi-pi* wavelength in nm'
    elif TASK_NAME == 'e_iso_n':
        label_col =  "E isomer n-pi* wavelength in nm"
    elif TASK_NAME == 'z_iso_n':
        label_col = 'Z isomer n-pi* wavelength in nm'
    elif TASK_NAME == 'thermal':
        label_col = 'rate of thermal isomerisation from Z-E in s-1'

    # Load data
    df = pd.read_csv(DATA_PATH)
    df = df.loc[~df[label_col].isna()]

    scaler = MinMaxScaler()
    scaler.fit(np.asarray(df[label_col]).reshape(-1, 1))

    # Create Train-Test Splits
    X_train, X_test = train_test_split(df, test_size=0.2)

    # Loads Mols from SMILES
    trainmols = [Chem.MolFromSmiles(m) for m in X_train['SMILES']]
    testmols = [Chem.MolFromSmiles(m) for m in X_test['SMILES']]

    # Initialse featurisers
    atom_featurizer = CanonicalAtomFeaturizer()
    n_feats = atom_featurizer.feat_size('h')
    print('Number of features: ', n_feats)

    # Create graphs and labels
    if GRAPH_TYPE == 'complete':
        train_g = [mol_to_complete_graph(m, node_featurizer=atom_featurizer) for m in trainmols]
        test_g = [mol_to_complete_graph(m, node_featurizer=atom_featurizer) for m in testmols]
    elif GRAPH_TYPE == 'bigraph':
        train_g = [mol_to_bigraph(m, node_featurizer=atom_featurizer) for m in trainmols]
        test_g = [mol_to_bigraph(m, node_featurizer=atom_featurizer) for m in testmols]

    train_y = torch.Tensor(scaler.transform(np.asarray(X_train[label_col]).reshape(-1,1)))
    test_y = torch.Tensor(scaler.transform(np.asarray(X_test[label_col]).reshape(-1, 1)))

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

    gcn_net = GCNPredictor(in_feats=n_feats,
                           hidden_feats=[64,32],
                           #activation=['relu', 'relu'],
                           batchnorm=[True, False],
                           dropout=[0.3,0],
                           classifier_hidden_feats=1
                           )
    gcn_net.to(device)

    loss_fn = MSELoss()
    optimizer = torch.optim.Adam(gcn_net.parameters(), lr=0.001)

    gcn_net.train()

    epoch_losses = []
    epoch_rmses = []
    epoch_pears = []
    #epoch_accuracies = []
    for epoch in range(1,501):
        epoch_loss = 0
        epoch_rmse = 0
        preds = []
        labs = []
        for i, (bg, labels) in enumerate(train_loader):
            labels = labels.to(device)
            atom_feats = bg.ndata.pop('h').to(device)
            atom_feats, labels = atom_feats.to(device), labels.to(device)
            y_pred = gcn_net(bg, atom_feats)
            labels = labels.unsqueeze(dim=1)
            loss = loss_fn(y_pred, labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            epoch_loss += loss.detach().item()

            #Inverse transform to get RMSE
            labels = scaler.inverse_transform(labels.reshape(-1, 1))
            y_pred = scaler.inverse_transform(y_pred.detach().numpy().reshape(-1, 1))
            #store labels and preds
            preds.append(y_pred)
            labs.append(labels)

        labs = np.concatenate(labs, axis=None)
        preds = np.concatenate(preds, axis=None)
        pearson, p = pearsonr(preds, labs)
        mae = np.divide(np.sum(np.abs(preds - labs)), len(train_data))
        rmse = np.sqrt(np.divide(np.sum(np.square(y_pred - labels)), len(train_data)))
        r2 = r2_score(preds, labs)

        epoch_loss /= (i + 1)
        if epoch % 20 == 0:
            print(f"epoch: {epoch}, LOSS: {epoch_loss:.3f}, RMSE: {rmse:.3f}, MAE: {mae:.3f}, R: {pearson:.3f}, R2: {r2:.3f}")
        epoch_losses.append(epoch_loss)
        epoch_rmses.append(rmse)

    # Evaluate
    gcn_net.eval()
    test_loss = 0
    squared_errors = []
    preds = []
    labs = []
    for i, (bg, labels) in enumerate(test_loader):
        labels = labels.to(device)
        atom_feats = bg.ndata.pop('h').to(device)
        atom_feats, labels = atom_feats.to(device), labels.to(device)
        y_pred = gcn_net(bg, atom_feats)
        labels = labels.unsqueeze(dim=1)

        #Inverse transform to get RMSE
        labels = scaler.inverse_transform(labels.reshape(-1, 1))
        y_pred = scaler.inverse_transform(y_pred.detach().numpy().reshape(-1, 1))

        preds.append(y_pred)
        labs.append(labels)

    labs = np.concatenate(labs, axis=None)
    preds = np.concatenate(preds, axis=None)

    pearson, p = pearsonr(preds, labs)
    mae = np.divide(np.sum(np.abs(preds - labs)), len(test_data))
    rmse = np.sqrt(np.divide(np.sum(np.square(y_pred - labels)), len(test_data)))
    r2 = r2_score(preds, labs)

    print(f'Test RMSE: {rmse:.3f}, MAE: {mae:.3f}, R: {pearson:.3f}, R2: {r2:.3f}')





