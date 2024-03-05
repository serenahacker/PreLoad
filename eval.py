import warnings
warnings.filterwarnings('ignore')
import torch
import numpy as np
from models import models, wideresnet, combinednet
from util import evaluation as evalutil
from util import gmm_utils
import util.dataloaders as dl
import util.misc as misc
import argparse
import os
from collections import defaultdict

parser = argparse.ArgumentParser()
parser.add_argument('--dataset', default='MNIST')
parser.add_argument('--ood_data', default='tiny300k', choices=['imagenet', 'tiny300k', 'uniform', 'smooth'])
args = parser.parse_args()

torch.manual_seed(9999)
np.random.seed(9999)

assert args.dataset in ['MNIST', 'FMNIST', 'CIFAR10', 'SVHN', 'CIFAR100']

# Note the order of the methods matters
methods = ['plain', 'ddu', 'noneclass', 'oe', 'preload', 'oe_finetuning', 'doe_ft', 'energy_finetuning', 'preload_ft'] 

pretrained_path = './pretrained_models'
path_ood =  ('/'+args.ood_data)

_, test_loader = dl.datasets_dict[args.dataset](train=False, augm_flag=False, val_size=1000)
targets = torch.cat([y for x, y in test_loader], dim=0).numpy()

num_classes = 100 if args.dataset == 'CIFAR100' else 10
depth = 16
widen_factor = 4
if args.dataset in ['MNIST', 'FMNIST']:
    size_of_Tx = 1024
else:
    size_of_Tx = 64 * widen_factor

ood_noise_names = ['UniformNoise', 'Noise', 'FarAway', 'FarAway2']
if args.ood_data == 'uniform':
    ood_noise_names = [ood_noise_names[1], ood_noise_names[2], ood_noise_names[3]]
elif args.ood_data == 'smooth':
    ood_noise_names = [ood_noise_names[0], ood_noise_names[2], ood_noise_names[3]]

ood_test_names = {
    'MNIST': ['FMNIST', 'EMNIST', 'KMNIST', 'GrayCIFAR10'],
    'FMNIST': ['MNIST', 'EMNIST', 'KMNIST', 'GrayCIFAR10'],
    'CIFAR10': ['SVHN', 'LSUN', 'CIFAR100', 'FMNIST3D'],
    'SVHN': ['CIFAR10', 'LSUN', 'CIFAR100', 'FMNIST3D'],
    'CIFAR100': ['SVHN', 'LSUN', 'CIFAR10', 'FMNIST3D'],
}

ood_names = ood_test_names[args.dataset] + ood_noise_names
ood_test_loaders = {}

for ood_name in ood_test_names[args.dataset]:
    ood_test_loaders[ood_name] = dl.datasets_dict[ood_name](train=False)

for ood_name in ood_noise_names:
    ood_test_loaders[ood_name] = dl.datasets_dict[ood_name](dataset=args.dataset, train=False, size=10000)

tab_mmc = defaultdict(lambda: defaultdict(list))
tab_auroc = defaultdict(lambda: defaultdict(list))
tab_auprc = defaultdict(lambda: defaultdict(list))
tab_fpr95 = defaultdict(lambda: defaultdict(list))
tab_acc = defaultdict(list)
tab_cal = defaultdict(list)


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
        
    method_name = 'plain' if method == 'ddu' else method
    if method_name != 'plain':
        path += path_ood
    path += f'/{args.dataset}_{method_name}_{seed}.pt'

    model.cuda()
    model.load_state_dict(torch.load(path))
    model.eval()

    return model


def predict_(model, method, test_dl, test_feature_loader):
    py = evalutil.predict(test_dl, model)
    return py.cpu().numpy()


@torch.no_grad()
def evaluate(method, seed, verbose=True):
    model = load_model(method, seed)
    USE_NONE_CLASS = method in ['noneclass', 'preload', 'preload_ft']

    py_in = predict_(model, method, test_loader, test_feature_loaders[seed-1])
    if method == 'ddu': 
        train_loader = dl.datasets_dict[args.dataset](train=True)
        embeddings, labels = gmm_utils.get_embeddings(model, train_loader, size_of_Tx, torch.double, 'cuda', 'cuda')
        gda, jitter_eps = gmm_utils.gmm_fit(embeddings, labels, num_classes)
        train_log_probs_B_Y, train_labels = gmm_utils.gmm_evaluate(model, gda, train_loader, 'cuda', num_classes, 'cuda')
        train_densities = torch.logsumexp(train_log_probs_B_Y, dim=-1)
        train_min_density = train_densities.min().item()

        in_densities = []
        for x, y in test_loader:
            x = x.cuda()
            log_probs_B_Y = gmm_utils.gmm_forward(model, gda, x)
            density = torch.logsumexp(log_probs_B_Y, dim=-1)-train_min_density
            in_densities.append(density)
        in_densities = torch.cat(in_densities)
    elif method == 'energy_finetuning':
        energy_in = evalutil.predict(test_loader, model, energy=True)

    acc = evalutil.get_acc(py_in, targets)
    ece, _ = evalutil.get_calib(py_in if not USE_NONE_CLASS else py_in[:, :-1], targets)
    mmc = evalutil.get_mmc(py_in, USE_NONE_CLASS)

    tab_mmc[method][args.dataset].append(mmc)
    tab_acc[method].append(acc)
    tab_cal[method].append(ece)

    if verbose:
        print(f'[In, {method}] Acc: {acc:.1f}; ECE: {ece:.1f}; MMC: {mmc:.3f}')

    for ood_name, ood_test_loader in ood_test_loaders.items():
        ood_test_feature_loader = misc.get_feature_loader(pretrained_plain_models[seed-1], ood_test_loader, batch_size=128, train=False)
        py_out = predict_(model, method, ood_test_loader, ood_test_feature_loader)

        mmc = evalutil.get_mmc(py_out, USE_NONE_CLASS)

        if method == 'ddu':
            out_densities = []
            for x, y in ood_test_loader:
                x = x.cuda()
                log_probs_B_Y = gmm_utils.gmm_forward(model, gda, x)
                density = torch.logsumexp(log_probs_B_Y, dim=-1)-train_min_density
                out_densities.append(density)
            out_densities = torch.cat(out_densities)
            py_in = in_densities.cpu().numpy()
            py_out = out_densities.cpu().numpy()
        elif method == 'energy_finetuning':
            energy_out = evalutil.predict(ood_test_loader, model, energy=True)
            py_in = -energy_in.cpu().numpy()
            py_out = -energy_out.cpu().numpy()

        auroc = evalutil.get_auroc(py_in, py_out, USE_NONE_CLASS, method == 'ddu', method == 'energy_finetuning')
        auprc = evalutil.get_aupr(py_in, py_out, USE_NONE_CLASS, method == 'ddu', method == 'energy_finetuning')
        fpr95, _ = evalutil.get_fpr95(py_in, py_out, USE_NONE_CLASS, method == 'ddu', method == 'energy_finetuning')

        tab_mmc[method][ood_name].append(mmc)
        tab_auroc[method][ood_name].append(auroc)
        tab_auprc[method][ood_name].append(auprc)
        tab_fpr95[method][ood_name].append(fpr95)

        if verbose:
            print(f'[Out-{ood_name}, {method}] MMC: {mmc:.1f}; AUROC: {auroc:.1f}; '
                + f'AUPRC: {auprc:.1f} FPR@95: {fpr95:.1f}')

    if verbose:
        print()

pretrained_plain_models = [load_model('plain', seed, feat_extractor=True) for seed in range(1, 6)]
test_feature_loaders = [misc.get_feature_loader(pretrained_model, test_loader, batch_size=128, train=False) for pretrained_model in pretrained_plain_models]

print('----------', args.dataset, args.ood_data, '----------')
for method in methods:
    print('Method:', method)
    for seed in range(1, 6):
        print('Seed:', seed)
        evaluate(method, seed)
    print()
    print('-------------------------------------------')
    print()

# Save results
dir_name = f'results/'
dir_name += f'{args.ood_data}/{args.dataset}'

if not os.path.exists(dir_name):
    os.makedirs(dir_name)

np.save(f'{dir_name}/mmc', dict(tab_mmc))
np.save(f'{dir_name}/auroc', dict(tab_auroc))
np.save(f'{dir_name}/auprc', dict(tab_auprc))
np.save(f'{dir_name}/fpr95', dict(tab_fpr95))
np.save(f'{dir_name}/acc', dict(tab_acc))
np.save(f'{dir_name}/cal', dict(tab_cal))
