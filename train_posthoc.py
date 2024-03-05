import torch
from torch import optim
from models.models import LeNetMadry
from models import posthocnet, wideresnet
from util.evaluation import *
import util.dataloaders as dl
import util.misc as misc
from tqdm import tqdm, trange
import numpy as np
import argparse
import os
from torch.cuda import amp
import json


parser = argparse.ArgumentParser()
parser.add_argument('--dataset', help='Pick one \\{"MNIST", "FMNIST", "CIFAR10", "SVHN", "CIFAR100"\\}', default='CIFAR10')
parser.add_argument('--ood_data', default='tiny300k', choices=['imagenet', 'tiny300k', 'smooth', 'uniform'])
parser.add_argument('--lmbd', type=float, default=1e-2)
parser.add_argument('--lr', type=float, default=5e-3)
parser.add_argument('--weight_decay', type=float, default=1e-4)
parser.add_argument('--randseed', type=int, default=1)
args = parser.parse_args()

np.random.seed(args.randseed)
torch.manual_seed(args.randseed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

print("------Arguments------")
print("dataset: ", args.dataset)
print("ood_data: ", args.ood_data)
print("lmbd: ", args.lmbd)
print("lr: ", args.lr)
print("weight_decay: ", args.weight_decay)
print("randseed: ", args.randseed)

pretrained_path = './pretrained_models'
plain_model_suffix = f'_plain_{args.randseed}'

assert args.dataset in ['MNIST', 'FMNIST', 'CIFAR10', 'SVHN', 'CIFAR100'], 'Invalid dataset.'
# Assert that the corresponding pretrained plain network is in the pretrained_models directory
assert os.path.isfile(f'{pretrained_path}/{args.dataset}{plain_model_suffix}.pt'), 'Plain model not pretrained.'

batch_size = 128
n_epochs = 10

path = f'./posthoc_models/{args.ood_data}'
model_suffix = f'_{args.randseed}'

train_loader = dl.datasets_dict[args.dataset](train=True, batch_size=batch_size)
val_loader, test_loader = dl.datasets_dict[args.dataset](train=False, augm_flag=False, val_size=1000)
val_targets = torch.cat([y for x, y in val_loader], dim=0).numpy()
test_targets = torch.cat([y for x, y in test_loader], dim=0).numpy()
# number of classes not including the noneclass
num_classes = 100 if args.dataset == 'CIFAR100' else 10

print(len(train_loader.dataset), len(test_loader.dataset))

# Load the pretrained
if args.dataset in ['MNIST', 'FMNIST']:
    pretrained_model = LeNetMadry(num_classes, feature_extractor=True)
    size_of_Tx = 1024
else:
    depth = 16
    widen_factor = 4
    pretrained_model = wideresnet.WideResNet(depth, widen_factor, num_classes, feature_extractor=True)
    size_of_Tx = 64 * widen_factor

pretrained_model.load_state_dict(torch.load(f'{pretrained_path}/{args.dataset}{plain_model_suffix}.pt'))
pretrained_model.cuda()
pretrained_model.eval()
for p in pretrained_model.parameters():
    p.requires_grad = False

print("size of Tx: ", size_of_Tx)
feature_loader = misc.get_feature_loader(pretrained_model, train_loader, batch_size=batch_size, train=True)
val_feature_loader =  misc.get_feature_loader(pretrained_model, val_loader, batch_size=batch_size, train=False)
test_feature_loader = misc.get_feature_loader(pretrained_model, test_loader, batch_size=batch_size, train=False)

model = posthocnet.PostHocNet(size_of_Tx, num_classes)

# read sweep params from json file
sweep_params_file = f'hyperparameter_sweeps/posthoc_best_hyperparams.json'
with open(sweep_params_file, 'r') as f:
    sweep_params = json.load(f)
SWEPT = args.ood_data in sweep_params and args.dataset in sweep_params[args.ood_data]
if SWEPT:
    best_hyperparams = sweep_params[args.ood_data][args.dataset]
    print("Using params from sweep: ", best_hyperparams)
    opt = optim.Adam(model.parameters(), lr=best_hyperparams['lr'], weight_decay=best_hyperparams['weight_decay'])
else: 
    opt = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

print(f'Num. params: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}')

model.cuda()
model.train()

criterion = torch.nn.CrossEntropyLoss(reduction='none')

## T_max is the max iterations: n_epochs x n_batches_per_epoch
scheduler = optim.lr_scheduler.CosineAnnealingLR(opt, T_max=n_epochs*len(feature_loader))
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
        ood_loader = dl.UniformNoise(train=True, dataset=args.dataset, size=len(feature_loader.dataset), batch_size=batch_size)

    data_iter = enumerate(zip(feature_loader, ood_loader))
    train_loss = 0

    for batch_idx, data in data_iter:
        model.train()
        opt.zero_grad()

        (Tx_in, wTx_in, y_in), (x_out, _) = data
        m = len(Tx_in)  # Batch size
        x_out = x_out[:m]  # To ensure the same batch size
        # Last class, zero-indexed
        y_out = (num_classes) * torch.ones(m).long()
        Tx_in, wTx_in, y_in = Tx_in.cuda(non_blocking=True), wTx_in.cuda(non_blocking=True), y_in.long().cuda(non_blocking=True)
        x_out, y_out = x_out.cuda(non_blocking=True), y_out.cuda(non_blocking=True)
        y = torch.cat([y_in, y_out], dim=0)

        with amp.autocast():
            with torch.no_grad():
                Tx_out, wTx_out = pretrained_model(x_out)
            Tx = torch.cat([Tx_in, Tx_out], dim=0)
            wTx = torch.cat([wTx_in, wTx_out], dim=0)
            outputs = model(Tx, wTx).squeeze()

            loss = criterion(outputs, y)

            # If the datapoint is a perturbed point, multiply the loss by lambda
            if SWEPT:
                loss = torch.where(y != num_classes, loss, best_hyperparams['lmbd']*loss)
            else:
                loss = torch.where(y != num_classes, loss, args.lmbd*loss)

            loss = loss.mean()

        scaler.scale(loss).backward()
        # Note (from https://pytorch.org/docs/stable/notes/amp_examples.html):
        # scaler.step() first unscales the gradients of the optimizer's assigned params.
        # If these gradients do not contain infs or NaNs, optimizer.step() is then called,
        # otherwise, optimizer.step() is skipped.
        scaler.step(opt)
        scaler.update()
        scheduler.step()

        train_loss = 0.9*train_loss + 0.1*loss.item()

    model.eval()
    pred = predict_posthoc(test_feature_loader, model).cpu().numpy()
    acc_test = np.mean(np.argmax(pred, 1) == test_targets)*100
    mmc_test = pred.max(-1).mean()*100
    pred = predict_posthoc(val_feature_loader, model).cpu().numpy()
    acc_val = np.mean(np.argmax(pred, 1) == val_targets)*100
    mmc_val = pred.max(-1).mean()*100

    pbar.set_description(
        f'[Epoch: {epoch+1}; loss: {train_loss:.3f}; acc: {acc_test:.1f}; mmc: {mmc_test:.1f}]'
    )

if not os.path.exists(path):
    os.makedirs(path)

torch.save(model.state_dict(), f'{path}/{args.dataset}{model_suffix}.pt')

## Try loading and testing
model.load_state_dict(torch.load(f'{path}/{args.dataset}{model_suffix}.pt'))
model.eval()

print()

## In-distribution
py_in = predict_posthoc(test_feature_loader, model).cpu().numpy()
acc_in = np.mean(np.argmax(py_in, 1) == test_targets)*100
print(f'Accuracy: {acc_in:.1f}')
