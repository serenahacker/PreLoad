import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset


def get_one_hot(targets, nb_classes, torch=False):
    res = np.eye(nb_classes)[np.array(targets).reshape(-1)]
    onehot = res.reshape(list(targets.shape)+[nb_classes])

    if torch:
        onehot = torch.tensor(onehot).float()

    return onehot

def get_feature_loader(pretrained_model, data_loader, batch_size=128, train=True):
    Txs = []
    wTxs = []
    targets = []

    with torch.no_grad():
        for (x, y) in data_loader:
            x = x.cuda()
            Tx, wTx = pretrained_model(x)
            Tx, wTx = Tx.cpu(), wTx.cpu()
            Txs.append(Tx)
            wTxs.append(wTx)
            targets.append(y)

    Txs = torch.cat(Txs, dim=0)
    wTxs = torch.cat(wTxs, dim=0)
    targets = torch.cat(targets, dim=0)

    feature_loader = DataLoader(
        TensorDataset(Txs, wTxs, targets), batch_size=batch_size, shuffle=train
    )
    return feature_loader
