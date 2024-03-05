import numpy as np
import argparse
import pandas as pd
import scipy.stats as st

parser = argparse.ArgumentParser()
parser.add_argument('--type', default='auroc', choices=['auroc', 'auprc', 'fpr95', 'mmc', 'acc', 'cal'])
parser.add_argument('--ood_data', default='tiny300k', choices=['imagenet', 'tiny300k', 'uniform', 'smooth'])
parser.add_argument('--averaged', default=False)
args = parser.parse_args()

path = f'./results/{args.ood_data}/'

# Note: the order of the methods matters
methods = ['plain', 'ddu', 'noneclass', 'oe', 'preload', 'oe_finetuning', 'doe_ft', 'energy_finetuning', 'preload_ft']  
datasets = [
    'MNIST', 
    'FMNIST', 
    'SVHN', 
    'CIFAR10', 
    'CIFAR100'
]

test_dsets = {
    'MNIST': ['FMNIST', 'EMNIST', 'KMNIST', 'GrayCIFAR10'],
    'FMNIST': ['MNIST', 'EMNIST', 'KMNIST', 'GrayCIFAR10'],
    'CIFAR10': ['SVHN', 'LSUN', 'CIFAR100', 'FMNIST3D'],
    'SVHN': ['CIFAR10', 'LSUN', 'CIFAR100', 'FMNIST3D'],
    'CIFAR100': ['SVHN', 'LSUN', 'CIFAR10', 'FMNIST3D'],
}

if args.ood_data == 'smooth':
    for dataset in datasets:
        test_dsets[dataset] += ['UniformNoise', 'FarAway', 'FarAway2']
elif args.ood_data == 'uniform':
    for dataset in datasets:
        test_dsets[dataset] += ['Noise', 'FarAway', 'FarAway2']
else:
    for dataset in datasets:
        test_dsets[dataset] += ['UniformNoise', 'Noise', 'FarAway', 'FarAway2']

method_strs = {
    'plain': 'Standard', 
    'ddu': 'DDU',
    'noneclass': 'NC', 
    'oe': 'OE', 
    'preload': 'PreLoad',
    'oe_finetuning': 'OE-FT',
    'doe_ft': 'DOE-FT',
    'energy_finetuning': 'Energy-FT',
    'preload_ft': 'PreLoad-FT'
}

dataset_strs = {
    'MNIST': 'MNIST', 'FMNIST': 'F-MNIST', 'CIFAR10': 'CIFAR-10',
    'SVHN': 'SVHN', 'CIFAR100': 'CIFAR-100', 'EMNIST': 'E-MNIST', 'KMNIST': 'K-MNIST',
    'UniformNoise': 'Uniform', 'FarAway': 'FarAway', 'FarAway2': 'FarAway-RD', 
    'Noise': 'Smooth', 'GrayCIFAR10': 'CIFAR-Gr', 'LSUN': 'LSUN-CR', 'FMNIST3D': 'FMNIST-3D'
}

acc_or_cal = args.type in ['acc', 'cal']

table_means = {method: [] for method in methods}
table_stds = {method: [] for method in methods}

# Add the table header
print('\\begin{table}')
print(f'\\caption{{{args.type.upper()}{" averaged" if args.averaged else ""}, {args.ood_data}}}')
if acc_or_cal:
    print('\\begin{tabular}{l ' + 'c '*len(datasets) + '}')
    print('\n\\midrule\n')
    str = '& '
    str += ' & '.join([f'\\textbf{{{dataset_strs[dset]}}}' for dset in datasets])
else:
    print('\\begin{tabular}{l ' + 'c '*len(methods) + '}')
    print('\n\\midrule\n')
    str = '\\textbf{Datasets} & '
    str += ' & '.join([f'\\textbf{{{method_strs[method]}}}' for method in methods])
str += ' \\\\'
print(str)
print('\n\\midrule\n')


for i, dset in enumerate(datasets):
    fname = f'{path+dset}'
    vals = np.load(f'{fname}/{args.type}.npy', allow_pickle=True).item()
    vals = pd.DataFrame(vals)
    vals = vals.drop(columns=[col for col in vals.columns if col not in methods])

    if args.type not in ['acc', 'mmc', 'cal']:
        vals = vals.drop(index=[idx for idx in vals.index if idx not in test_dsets[dset]])
    if args.averaged:
        # Don't include either FarAway or FarAway2 (or dset for MMC) in the averages
        vals = vals.drop(index=[idx for idx in vals.index if idx in ['FarAway', 'FarAway2', dset]])

    if not acc_or_cal or args.averaged:
        vals = pd.DataFrame(vals).transpose()
    
        means = {}
        stds = {}
        
        for col in vals:
            means[col] = [np.mean(val) for val in vals[col].values]
            stds[col] = [st.sem(val) for val in vals[col].values]

        combined_vals = vals.apply(lambda row: sum(row, []), axis=1).apply(pd.Series)
        df_means = vals.copy()
        df_stds = vals.copy()
        for col in vals:
            df_means[col] = means[col]
            df_stds[col] = stds[col]

    def print_bold(dset_name, means, stds, mark_bold=False):
        if mark_bold:
            means = [round(m, 1) for m in means]
            top_means = np.max(means) if args.type in ['auroc', 'auprc'] else np.min(means)
            tops = np.argwhere(means == top_means).flatten()
            bolds = [True if j in tops else False for j, _ in enumerate(means)]
        else:
            bolds = [False]*len(means)

        str = f'{dataset_strs[dset_name]} & '
        str += ' & '.join([
            f'\\textbf{{{m:.1f}}}$\\pm${s:.1f}' if bold else f'{m:.1f}$\\pm${s:.1f}'
            for m, s, bold in zip(means, stds, bolds)
        ])
        str += ' \\\\'
        print(str)

    if args.averaged:
        combined_vals_means = pd.DataFrame(combined_vals).transpose().mean(axis=0)
        combined_vals_std_errs = combined_vals.apply(st.sem, axis=1)
        str = f'\\textbf{{{dataset_strs[dset]}}} & '
        str += ' & '.join([f'{combined_vals_means[m]:.1f}$\\pm${combined_vals_std_errs[m]:.1f}' for m in methods])
        str += ' \\\\'
        print(str)
        print('\n\\midrule')
    elif not acc_or_cal:
        # Print LaTex code
        if args.type in ['auroc', 'auprc', 'fpr95']:
            str = f'\\textbf{{{dataset_strs[dset]}}} '
            str += '& '*len(methods)
            str += '\\\\'
            print(str)
        else:  # MMC
            str = f'\\textbf{{{dataset_strs[dset]}}} & '
            str += ' & '.join([f'{v_:.1f}' for v_ in means[dset]])
            str += ' \\\\'
            print(str)

        for k in df_means.keys():
            if args.type == 'mmc' and k == dset:
                continue
            if args.ood_data == 'smooth' and k == 'Noise':
                continue
            print_bold(k, df_means[k], df_stds[k])

        if i < len(datasets):
            print('\n\\midrule\n')
    else:  # acc_or_cal
        for method in methods:
            table_means[method].append(vals[method].mean())
            table_stds[method].append(vals[method].sem())

if acc_or_cal:
    for i, method in enumerate(methods):
        val_str = ' & '.join([f'{m:.1f}' for m in table_means[method]])
        print(f'{method_strs[method]} & {val_str} \\\\')

        print('\n\\midrule')


# Add the table footer
print('\\end{tabular}')
print('\\end{table}')