import numpy as np
import pickle
import os, sys, argparse
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from math import *
import tqdm
import seaborn as sns
from util.plot_helper import get_mpl_rcParams


parser = argparse.ArgumentParser()
parser.add_argument('--ood_dset', default='tiny300k', choices=['tiny300k', 'imagenet', 'uniform', 'smooth'])
args = parser.parse_args()

path = f'./results/{args.ood_dset}/MNIST-R'

methods = ['plain', 'noneclass', 'oe', 'oe_finetuning', 'doe_ft', 'energy_finetuning', 'preload', 'preload_ft']

method_strs = {
    'plain': 'Standard', 
    'noneclass': 'NC', 
    'oe': 'OE', 
    'preload': 'PreLoad',
    'oe_finetuning': 'OE-FT',
    'doe_ft': 'DOE-FT',
    'energy_finetuning': 'Energy-FT',
    'preload_ft': 'PreLoad-FT'
}
x = list(range(0, 181, 15))

def load(metric='mmc'):
    return np.load(f'{path}/{metric}.npy', allow_pickle=True).item()

def plot(vals, metric, legend=False):
    FIG_WIDTH = 0.5  # i.e. 100% of paper's width
    FIG_HEIGHT = 0.2  # i.e. 15% of paper's height
    rc_params, fig_width, fig_height = get_mpl_rcParams(FIG_WIDTH, FIG_HEIGHT)
    plt.rcParams.update(rc_params)

    fig, ax = plt.subplots(1, 1, constrained_layout=True)
    fig.set_size_inches(fig_width, fig_height)

    for method in methods:
        v = vals[method]
        y = [v[angle][0] for angle in x]
        ax.plot(x, y, label=method_strs[method], alpha=1)

    ticks = range(0, 181, 30)
    ax.set_xticks(ticks)
    ax.set_xlim(0, 180)
    ax.set_xlabel('Rotation Angle')
    ax.set_ylabel(metric.upper())
    ax.set_title('Rotated MNIST')

    if metric != 'loglik':
        ax.set_ylim(bottom=0)

    if metric in ['mmc', 'acc', 'aur']:
        ax.set_ylim(top=1)

    if legend:
        ax.legend(ncol=2, loc='lower right')

    dir_name = f'figs/MNIST-R/{args.ood_dset}'
    if not os.path.exists(dir_name):
        os.makedirs(dir_name)
    plt.savefig(f'{dir_name}/mnistr_{metric}.pdf')


plot(load('loglik'), 'loglik', legend=True)
plot(load('brier'), 'brier', legend=True)
plot(load('ece'), 'ece', legend=True)
