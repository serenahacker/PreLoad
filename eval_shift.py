import warnings
warnings.filterwarnings('ignore')
import torch
import numpy as np
from models import models, wideresnet, combinednet
from util import evaluation as evalutil
import util.dataloaders as dl
import util.misc as misc
import argparse
import os
import json
from collections import defaultdict
from torchvision.transforms import functional as TF
import torch.utils.data as data_utils

parser = argparse.ArgumentParser()
parser.add_argument('--dataset', default='MNIST', choices=['MNIST', 'CIFAR10'])
parser.add_argument('--ood_data', default='tiny300k', choices=['imagenet', 'tiny300k', 'uniform', 'smooth'])
args = parser.parse_args()

torch.manual_seed(9999)
np.random.seed(9999)

methods = ['plain', 'noneclass', 'oe', 'preload', 'oe_finetuning', 'doe_ft', 'energy_finetuning', 'preload_ft']

pretrained_path = './pretrained_models'
path_ood =  ('/'+args.ood_data)

_, test_loader = dl.datasets_dict[args.dataset](train=False, augm_flag=False, val_size=1000)

num_classes = 10  # Only MNIST & CIFAR10
depth = 16
widen_factor = 4

if args.dataset == 'MNIST':  # rotatedMNIST
    rotation_angles = range(15, 181, 15)
    translation_pixels = range(2, 15, 2)
    tab_acc = defaultdict(lambda: defaultdict(list))
    tab_mmc = defaultdict(lambda: defaultdict(list))
    tab_ece = defaultdict(lambda: defaultdict(list))
    tab_brier = defaultdict(lambda: defaultdict(list))
    tab_loglik = defaultdict(lambda: defaultdict(list))
else: # corruptedCIFAR10
    distortion_types = dl.CorruptedCIFAR10Dataset.distortions
    severity_levels = range(1, 6)  # 1 ... 5
    tab_acc = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
    tab_mmc = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
    tab_ece = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
    tab_brier = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
    tab_loglik = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))

def load_model(method, seed, feat_extractor=False):
    path = pretrained_path
    if 'preload' in method:
        model = combinednet.CombinedNet(args.dataset, num_classes)
    else:
        n_classes = num_classes+1 if method == 'noneclass' else num_classes
        if args.dataset in ['MNIST', 'FMNIST']:
            model = models.LeNetMadry(n_classes, feature_extractor=feat_extractor)
        else:
            model = wideresnet.WideResNet(depth, widen_factor, n_classes, feature_extractor=feat_extractor)
        
    if method != 'plain':
        path += path_ood
    path += f'/{args.dataset}_{method}_{seed}.pt'

    model.cuda()
    model.load_state_dict(torch.load(path))
    model.eval()

    return model


def predict_(model, method, test_dl, test_feature_loader):
    py = evalutil.predict(test_dl, model)
    return py.cpu().numpy()


@torch.no_grad()
def evaluate_rotMNIST(method, seed):
    model = load_model(method, seed)
    USE_NONE_CLASS = method in ['noneclass', 'preload', 'preload_ft']

    # Original test_loader
    py_in = predict_(model, method, test_loader, test_feature_loaders[seed-1])
    targets = torch.cat([y for _, y in test_loader], dim=0).numpy()

    py_in_full = py_in
    if USE_NONE_CLASS:
        py_in = py_in[:, :-1]  # Ignore the none class

    tab_acc[method][0].append(evalutil.get_acc(py_in, targets))
    tab_mmc[method][0].append(evalutil.get_mmc(py_in_full, USE_NONE_CLASS))
    tab_ece[method][0].append(evalutil.get_calib(py_in, targets)[0])
    # print("original")
    tab_brier[method][0].append(evalutil.get_brier(py_in, targets))
    tab_loglik[method][0].append(evalutil.get_loglik(py_in, targets))

    for angle in rotation_angles:
        # Rotate the test set
        X_shift = torch.cat([TF.rotate(x, angle) for x, _ in test_loader], dim=0)
        y_shift = torch.cat([y for _, y in test_loader], dim=0)
        shift_dset = data_utils.TensorDataset(X_shift, y_shift)
        shift_loader = data_utils.DataLoader(shift_dset, batch_size=128, pin_memory=True)

        shift_feature_loader = misc.get_feature_loader(
            pretrained_plain_models[seed-1], shift_loader, batch_size=128, train=False
        )
        py_shift = predict_(model, method, shift_loader, shift_feature_loader)

        py_shift_full = py_shift
        # print("before", py_shift.shape)
        if USE_NONE_CLASS:
            py_shift = py_shift[:, :-1]  # Ignore the none class
        # print("after", py_shift.shape)
        tab_acc[method][angle].append(evalutil.get_acc(py_shift, targets))
        tab_mmc[method][angle].append(evalutil.get_mmc(py_shift_full, USE_NONE_CLASS))
        tab_ece[method][angle].append(evalutil.get_calib(py_shift, targets)[0])
        # print("shift")
        tab_brier[method][angle].append(evalutil.get_brier(py_shift, targets))
        tab_loglik[method][angle].append(evalutil.get_loglik(py_shift, targets, reduction='mean'))


@torch.no_grad()
def evaluate_corrCIFAR10(method, seed):
    model = load_model(method, seed)
    USE_NONE_CLASS = method in ['noneclass', 'preload', 'preload_ft']

    for d in distortion_types:
        for s in severity_levels:
            # Rotate the test set
            shift_loader = dl.CorruptedCIFAR10(d, s)
            shift_feature_loader = misc.get_feature_loader(
                pretrained_plain_models[seed-1], shift_loader, batch_size=128, train=False
            )
            targets = torch.cat([y for _, y in shift_loader], dim=0).numpy()
            py_shift = predict_(model, method, shift_loader, shift_feature_loader)

            py_shift_full = py_shift
            if USE_NONE_CLASS:
                py_shift = py_shift[:, :-1]  # Ignore the none class

            tab_acc[method][d][s].append(evalutil.get_acc(py_shift, targets))
            tab_mmc[method][d][s].append(evalutil.get_mmc(py_shift_full, USE_NONE_CLASS))
            tab_ece[method][d][s].append(evalutil.get_calib(py_shift, targets)[0])
            tab_brier[method][d][s].append(evalutil.get_brier(py_shift, targets))
            tab_loglik[method][d][s].append(evalutil.get_loglik(py_shift, targets, reduction='mean'))

pretrained_plain_models = [load_model('plain', seed, feat_extractor=True) for seed in range(1, 6)]
test_feature_loaders = [misc.get_feature_loader(pretrained_model, test_loader, batch_size=128, train=False) for pretrained_model in pretrained_plain_models]

print('----------', args.dataset, args.ood_data, '----------')
for method in methods:
    print('Method:', method)
    for seed in range(1, 6):
        print('Seed:', seed)
        if args.dataset == 'MNIST':
            evaluate_rotMNIST(method, seed)
        else:
            evaluate_corrCIFAR10(method, seed)
    print()
    print('-------------------------------------------')
    print()

# Save results
shift_dataset_name = 'MNIST-R' if args.dataset == 'MNIST' else 'CIFAR10-C'
dir_name = f'results/{args.ood_data}/{shift_dataset_name}'

if not os.path.exists(dir_name):
    os.makedirs(dir_name)

if args.dataset == 'MNIST':
    np.save(f'{dir_name}/mmc.npy', dict(tab_mmc))
    np.save(f'{dir_name}/acc.npy', dict(tab_acc))
    np.save(f'{dir_name}/ece.npy', dict(tab_ece))
    np.save(f'{dir_name}/brier.npy', dict(tab_brier))
    np.save(f'{dir_name}/loglik.npy', dict(tab_loglik))
else:
    np.save(f'{dir_name}/mmc.npy', json.loads(json.dumps(tab_mmc)))
    np.save(f'{dir_name}/acc.npy', json.loads(json.dumps(tab_acc)))
    np.save(f'{dir_name}/ece.npy', json.loads(json.dumps(tab_ece)))
    np.save(f'{dir_name}/brier.npy', json.loads(json.dumps(tab_brier)))
    np.save(f'{dir_name}/loglik.npy', json.loads(json.dumps(tab_loglik)))