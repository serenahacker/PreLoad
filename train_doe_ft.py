# Used train function from https://github.com/QizhouWang/DOE/blob/main/doe_final.py and integrated with our codebase. 
import torch
import torch.nn.functional as F
from torch import optim
from models.models import LeNetMadry
from models import wideresnet
from util.awp import *
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
parser.add_argument('--ood_data', default='tiny300k', choices=['imagenet', 'tiny300k', 'smooth', 'uniform'])
parser.add_argument('--randseed', type=int, default=1)
parser.add_argument('--epochs', type=int, default=10, help='Number of epochs to train.')
parser.add_argument('--warmup', type=int, default=5)
args = parser.parse_args()

np.random.seed(args.randseed)
torch.manual_seed(args.randseed)
torch.cuda.manual_seed(args.randseed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

assert args.dataset in ['MNIST', 'FMNIST', 'CIFAR10', 'SVHN', 'CIFAR100'], 'Invalid dataset.'

batch_size = 128
path = './pretrained_models'

# Assert that the corresponding pretrained plain network is in the pretrained_models directory
assert os.path.isfile(f'{path}/{args.dataset}_plain_{args.randseed}.pt'), 'Plain model not pretrained.'

train_loader = dl.datasets_dict[args.dataset](train=True, batch_size=batch_size)
_, test_loader = dl.datasets_dict[args.dataset](train=False, augm_flag=False, val_size=1000)
test_targets = torch.cat([y for x, y in test_loader], dim=0).numpy()
num_classes = 100 if args.dataset == 'CIFAR100' else 10

print(len(train_loader.dataset), len(test_loader.dataset))

depth = 16
widen_factor = 4
if args.dataset in ['MNIST', 'FMNIST']:
    model = LeNetMadry(num_classes)
    opt = optim.Adam(model.parameters(), lr=1e-3, weight_decay=5e-4)
else:
    model = wideresnet.WideResNet(depth, widen_factor, num_classes)
    opt = optim.SGD(model.parameters(), lr=1e-3, momentum=0.9, weight_decay=5e-4, nesterov=True)

print(f'Num. params: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}')
model.cuda()
model.train()

## Load pretrained model
model.load_state_dict(torch.load(f'{path}/{args.dataset}_plain_{args.randseed}.pt'))
model.eval()

# criterion = torch.nn.CrossEntropyLoss(reduction='mean')

## T_max is the max iterations: args.epochs x n_batches_per_epoch
scheduler = optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs*len(train_loader))
pbar = trange(args.epochs)

## For automatic-mixed-precision
# scaler = amp.GradScaler()

if args.ood_data == 'imagenet':
    ood_loader = dl.ImageNet32(train=True, dataset=args.dataset, batch_size=batch_size)

diff = None

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

    if args.dataset in ['MNIST', 'FMNIST']:
        proxy = LeNetMadry(num_classes).cuda()
    else:
        proxy = wideresnet.WideResNet(depth, widen_factor, num_classes, dropRate = 0).cuda()
    proxy_optim = optim.SGD(proxy.parameters(), lr=1)
    model.train()

    for batch_idx, data in data_iter:
        (x_in, y_in), (x_out, _) = data
        m = len(x_in)  # Batch size
        x_out = x_out[:m]  # To ensure the same batch size

        x_in, y_in = x_in.cuda(non_blocking=True), y_in.long().cuda(non_blocking=True)
        x_out = x_out.cuda(non_blocking=True)
        x = torch.cat([x_in, x_out], dim=0)

        if epoch >= args.warmup:
            gamma = torch.Tensor([1e-2,1e-3,1e-4])[torch.randperm(3)][0]
            proxy.load_state_dict(model.state_dict())
            proxy.train()
            scale = torch.Tensor([1]).cuda().requires_grad_()
            proxy_x = proxy(x) * scale
            l_sur = (proxy_x[m:].mean(1) - torch.logsumexp(proxy_x[m:], dim=1)).mean()
            reg_sur = torch.sum(torch.autograd.grad(l_sur, [scale], create_graph = True)[0] ** 2)
            proxy_optim.zero_grad()
            reg_sur.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1)
            proxy_optim.step()
            if epoch == args.warmup and batch_idx == 0:
                diff = diff_in_weights(model, proxy)
            else:
                diff = average_diff(diff, diff_in_weights(model, proxy), beta = .6)
            add_into_weights(model, diff, coeff = gamma)

        model_x = model(x)
        l_ce = F.cross_entropy(model_x[:m], y_in)
        l_oe = - (model_x[m:].mean(1) - torch.logsumexp(model_x[m:], dim=1)).mean()
        if epoch >= args.warmup:
            loss = l_oe
        else: 
            loss = l_ce +  l_oe

        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1) 
        opt.step()

        if epoch >= args.warmup:
            add_into_weights(model, diff, coeff = - gamma)
            opt.zero_grad()
            model_x = model(x)
            l_ce = F.cross_entropy(model_x[:m], y_in)
            loss = l_ce
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1)
            opt.step()


        scheduler.step()
        train_loss = 0.9*train_loss + 0.1*loss.item()

    model.eval()
    pred = predict(test_loader, model).cpu().numpy()
    acc_test = np.mean(np.argmax(pred, 1) == test_targets)*100
    mmc_test = pred.max(-1).mean()*100

    pbar.set_description(
        f'[Epoch: {epoch+1}; loss: {train_loss:.3f}; acc: {acc_test:.1f}; mmc: {mmc_test:.1f}]'
    )

save_path = f'{path}/{args.ood_data}'
if not os.path.exists(save_path):
    os.makedirs(save_path)

model_suffix = f'doe_ft_{args.randseed}'
torch.save(model.state_dict(), f'{save_path}/{args.dataset}_{model_suffix}.pt')

## Try loading and testing
model.load_state_dict(torch.load(f'{save_path}/{args.dataset}_{model_suffix}.pt'))
model.eval()

## In-distribution
py_in = predict(test_loader, model).cpu().numpy()
acc_in = np.mean(np.argmax(py_in, 1) == test_targets)*100
print(f'Accuracy: {acc_in:.1f}')