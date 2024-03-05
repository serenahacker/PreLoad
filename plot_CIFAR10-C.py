import numpy as np
from collections import defaultdict
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os, sys, argparse
from util.plot_helper import get_mpl_rcParams


parser = argparse.ArgumentParser()
parser.add_argument('--ood_dset', default='tiny300k', choices=['tiny300k', 'imagenet', 'smooth'])
args = parser.parse_args()

methods = ['plain', 'noneclass', 'oe', 'oe_finetuning', 'doe_ft', 'energy_finetuning', 'preload', 'preload_ft']
metric2str = {'acc': 'Acc.', 'mmc': 'MMC', 'ece': 'ECE',
              'brier': 'Brier', 'loglik': 'LogLik'}
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

path = f'results/{args.ood_dset}/CIFAR10-C'
N = 10000  # n test points


def plot(metric='ece'):
    FIG_WIDTH = 0.5  # i.e. 100% of paper's width
    FIG_HEIGHT = 0.2  # i.e. 15% of paper's height
    rc_params, fig_width, fig_height = get_mpl_rcParams(FIG_WIDTH, FIG_HEIGHT)
    plt.rcParams.update(rc_params)

    fig, ax = plt.subplots(1, 1, constrained_layout=True)
    fig.set_size_inches(fig_width, fig_height)

    ax.set_title('Corrupted CIFAR10')

    metric_str = metric2str[metric]
    data = {'Method': [], 'Severity': [], metric_str: []}
    vals = np.load(f'{path}/{metric}.npy', allow_pickle=True).item()

    for method in methods:
        for distortion in vals[method].keys():
            if distortion == 'clean':
                continue

            for severity in vals[method][distortion].keys():
                data['Method'].append(method_strs[method])
                data['Severity'].append(int(severity))

                val = vals[method][distortion][severity][0]

                if metric == 'loglik':
                    val /= -N

                data[metric_str].append(val)


    df = pd.DataFrame(data)
    p = sns.boxplot(
        data=df, x='Severity', y=metric_str, hue='Method', fliersize=0,
        linewidth=0.5, ax=ax
    )
    ax.legend([],[], frameon=False)

    dir_name = f'figs/CIFAR10-C/{args.ood_dset}'
    if not os.path.exists(dir_name):
        os.makedirs(dir_name)
    plt.savefig(f'{dir_name}/cifar10c_{metric}.pdf')


plot(metric='loglik')
plot(metric='ece')
plot(metric='brier')
plot(metric='mmc')
