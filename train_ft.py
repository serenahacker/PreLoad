import torch
import torch.nn.functional as F
from torch import optim
from models.models import LeNetMadry
from models import wideresnet
from util.evaluation import *
import util.dataloaders as dl
from tqdm import tqdm, trange
import numpy as np
import argparse
import os
from torch.cuda import amp
import json


parser = argparse.ArgumentParser()
parser.add_argument('--dataset', help='Pick one \\{"MNIST", "FMNIST", "CIFAR10", "SVHN", "CIFAR100"\\}', default='CIFAR10')
parser.add_argument('--method', choices=['oe', 'energy'], default='oe')
parser.add_argument('--ood_data', default='tiny300k', choices=['imagenet', 'tiny300k', 'smooth', 'uniform'])
# m_in and m_out are the margin parameters for the energy method
parser.add_argument('--m_in', type=float, default=-25.)
parser.add_argument('--m_out', type=float, default=-7.)
parser.add_argument('--randseed', type=int, default=1)
args = parser.parse_args()

np.random.seed(args.randseed)
torch.manual_seed(args.randseed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

assert args.dataset in ['MNIST', 'FMNIST', 'CIFAR10', 'SVHN', 'CIFAR100'], 'Invalid dataset.'

batch_size = 128
n_epochs = 10
path = './pretrained_models'

# Assert that the corresponding pretrained plain network is in the pretrained_models directory
assert os.path.isfile(f'{path}/{args.dataset}_plain_{args.randseed}.pt'), 'Plain model not pretrained.'

train_loader = dl.datasets_dict[args.dataset](train=True, batch_size=batch_size)
val_loader, test_loader = dl.datasets_dict[args.dataset](train=False, augm_flag=False, val_size=1000)
val_targets = torch.cat([y for x, y in val_loader], dim=0).numpy()
test_targets = torch.cat([y for x, y in test_loader], dim=0).numpy()
num_classes = 100 if args.dataset == 'CIFAR100' else 10

print(len(train_loader.dataset), len(test_loader.dataset))

if args.dataset in ['MNIST', 'FMNIST']:
    model = LeNetMadry(num_classes)
    opt = optim.Adam(model.parameters(), lr=1e-3, weight_decay=5e-4)
else:
    depth = 16
    widen_factor = 4
    model = wideresnet.WideResNet(depth, widen_factor, num_classes)
    opt = optim.SGD(model.parameters(), lr=1e-3, momentum=0.9, weight_decay=5e-4, nesterov=True)

# read sweep params from json file
sweep_params_file = f'hyperparameter_sweeps/energy_ft_best_hyperparams.json'
with open(sweep_params_file, 'r') as f:
    sweep_params = json.load(f)
SWEPT = args.ood_data in sweep_params and args.dataset in sweep_params[args.ood_data]
if SWEPT:
    best_hyperparams = sweep_params[args.ood_data][args.dataset]
    print("Using params from sweep: ", best_hyperparams)

print(f'Num. params: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}')
model.cuda()
model.train()

## Load pretrained model
model.load_state_dict(torch.load(f'{path}/{args.dataset}_plain_{args.randseed}.pt'))
model.eval()

criterion = torch.nn.CrossEntropyLoss(reduction='mean')

## T_max is the max iterations: n_epochs x n_batches_per_epoch
scheduler = optim.lr_scheduler.CosineAnnealingLR(opt, T_max=n_epochs*len(train_loader))
pbar = trange(n_epochs)

## For automatic-mixed-precision
scaler = amp.GradScaler()

if args.ood_data == 'imagenet':
    ood_loader = dl.ImageNet32(train=True, dataset=args.dataset, batch_size=batch_size)

for epoch in pbar:
    # Get new data every epoch to avoid overfitting to noises
    if args.ood_data == 'imagenet':
        # Induce a randomness in the OOD batch since num_ood_data >> num_indist_data
        # The shuffling of ood_loader only happens when all batches are already yielded
        ood_loader.dataset.offset = np.random.randint(len(ood_loader.dataset))
    elif args.ood_data == 'tiny300k':
        ood_loader = dl.Tiny300k(dataset=args.dataset, batch_size=batch_size)
    elif args.ood_data == 'smooth':
        ood_loader = dl.Noise(train=True, dataset=args.dataset, batch_size=batch_size)
    elif args.ood_data == 'uniform':
        ood_loader = dl.UniformNoise(train=True, dataset=args.dataset, size=len(train_loader.dataset), batch_size=batch_size)

    data_iter = enumerate(zip(train_loader, ood_loader))
    train_loss = 0

    for batch_idx, data in data_iter:
        model.train()
        opt.zero_grad()

        (x_in, y_in), (x_out, _) = data
        m = len(x_in)  # Batch size
        x_out = x_out[:m]  # To ensure the same batch size

        x_in, y_in = x_in.cuda(non_blocking=True), y_in.long().cuda(non_blocking=True)
        x_out = x_out.cuda(non_blocking=True)
        x = torch.cat([x_in, x_out], dim=0)

        with amp.autocast():
            outputs = model(x).squeeze()
            if args.method == 'oe':
                loss = criterion(outputs[:m], y_in)
                # Sum all log-probs directly (0.5 following Hendrycks)
                loss += -0.5*1/num_classes*torch.log_softmax(outputs[m:], 1).mean()
            elif args.method == 'energy':
                loss = criterion(outputs[:m], y_in)
                Ec_in = -torch.logsumexp(outputs[:m], dim=1)
                Ec_out = -torch.logsumexp(outputs[m:], dim=1)
                if SWEPT:
                    loss += 0.1*(torch.pow(F.relu(Ec_in-best_hyperparams['m_in']), 2).mean() + torch.pow(F.relu(best_hyperparams['m_out']-Ec_out), 2).mean())
                else: 
                    loss += 0.1*(torch.pow(F.relu(Ec_in-args.m_in), 2).mean() + torch.pow(F.relu(args.m_out-Ec_out), 2).mean())

        scaler.scale(loss).backward()
        scaler.step(opt)
        scaler.update()
        scheduler.step()

        train_loss = 0.9*train_loss + 0.1*loss.item()

    model.eval()
    pred = predict(test_loader, model).cpu().numpy()
    acc_test = np.mean(np.argmax(pred, 1) == test_targets)*100
    mmc_test = pred.max(-1).mean()*100
    pred = predict(val_loader, model).cpu().numpy()
    acc_val = np.mean(np.argmax(pred, 1) == val_targets)*100
    mmc_val = pred.max(-1).mean()*100

    pbar.set_description(
        f'[Epoch: {epoch+1}; loss: {train_loss:.3f}; acc: {acc_test:.1f}; mmc: {mmc_test:.1f}]'
    )

save_path = f'{path}/{args.ood_data}'
if not os.path.exists(save_path):
    os.makedirs(save_path)

model_suffix = f'{args.method}_finetuning_{args.randseed}'
torch.save(model.state_dict(), f'{save_path}/{args.dataset}_{model_suffix}.pt')

## Try loading and testing
model.load_state_dict(torch.load(f'{save_path}/{args.dataset}_{model_suffix}.pt'))
model.eval()

print()

## In-distribution
py_in = predict(test_loader, model).cpu().numpy()
acc_in = np.mean(np.argmax(py_in, 1) == test_targets)*100
print(f'Accuracy: {acc_in:.1f}')
