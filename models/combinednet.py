import torch
import torch.nn as nn
from models.models import LeNetMadry
from models import wideresnet, posthocnet


class CombinedNet(nn.Module):

    def __init__(self, dataset, num_classes):
        super().__init__()
        if dataset in ['MNIST', 'FMNIST']:
            self.network_part1 = LeNetMadry(num_classes, feature_extractor=True)
            size_of_Tx = 1024
        else: 
            depth = 16
            widen_factor = 4
            self.network_part1 = wideresnet.WideResNet(depth, widen_factor, num_classes, feature_extractor=True)
            size_of_Tx = 64 * widen_factor
        
        self.network_part2 = posthocnet.PostHocNet(size_of_Tx, num_classes)

    def forward(self, x):
        Tx, wTx = self.network_part1(x)
        x = self.network_part2(Tx, wTx)
        return x
