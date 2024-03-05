import torch
import torch.nn as nn


class PostHocNet(nn.Module):

    def __init__(self, size_of_Tx, num_classes):
        super().__init__()
        self.fc = nn.Linear(size_of_Tx, 1)
        self.size_of_Tx = size_of_Tx
        self.num_classes = num_classes

    def forward(self, Tx, wTx):
        # Classes 1..k:
        # w_y^T T(x)+b_y
        p = wTx

        # Class k+1:
        # e^(w_z)^T (T(x))^2 + b_z
        q = (torch.matmul(torch.square(Tx), torch.transpose(torch.exp(self.fc.weight), 0, 1))+self.fc.bias)

        # Concatenate classes 1...k with class k+1
        z = torch.cat((p, q), 1)
        return z
