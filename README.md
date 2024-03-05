# PreLoad

Code for the paper titled "Preventing Arbitrarily High Confidence on Far-Away Data in
Point-Estimated Discriminative Neural Networks".

```
@inproceedings{rashid2024preload,
  title={Preventing Arbitrarily High Confidence on Far-Away Data in Point-Estimated Discriminative Neural Networks},
  author={Rashid, Ahmad and Hacker, Serena and Zhang, Guojun and Kristiadi, Agustinus and Poupart, Pascal},
  booktitle={AISTATS},
  year={2024}
}
```

## Setting up:

1. Install PyTorch and TorchVision (<https://pytorch.org/get-started/locally/>).
2. Run: `conda create --name ENV_NAME --file conda_env.txt`.
3. Then: `conda activate ENV_NAME`.

## Download Datasets:

1. Check the dataset `path` in `util/dataloaders.py`, line 27.
2. Let's assume it's `~/Datasets`. Then:
3. `cd ~/Datasets`
4. Download 300k random images dataset. This OOD training set is an alternative to the now-taken-down 80M Tiny Images dataset. See <https://github.com/hendrycks/outlier-exposure/tree/master>. Run `wget https://people.eecs.berkeley.edu/~hendrycks/300K_random_images.npy`.
5. If you would like to use ImageNet as an OOD training set, download downsampled ImageNet from http://image-net.org/download-images (Train(32x32) and Val(32x32)) into `'~/Datasets/Imagenet32'` directory. Unzip `'~/Datasets/Imagenet32/Imagenet32_train.zip'` and `'~/Datasets/Imagenet32/Imagenet32_val.zip'`. Note: this step is not needed in order to reproduce the paper's results as we show results for 300k random images as the OOD training set.
6. Download LSUN classroom dataset (used for evaluation in results tables) by cloning https://github.com/fyu/lsun and running `python3 download.py -c classroom -o ~/Datasets`. Unzip `classroom_train_lmdb.zip` and `classroom_val_lmdb.zip` into `~/Datasets`.
7. Download the CIFAR-10-C dataset (used for dataset shift experiment in calibration figures) from https://zenodo.org/records/2535967. Download `CIFAR-10-C.tar` and unzip it into `~/Datasets`.

## Training "Trained from Scratch" Baselines:

To train a single "trained from scratch" baseline model, run `python train.py --dataset {MNIST, FMNIST, CIFAR10, SVHN, CIFAR100} --method {plain, oe, noneclass} --ood_data {tiny300k, imagenet, smooth, uniform} --randseed {r}`, where method can be Standard = `plain`, OE = `oe`, or NC = `noneclass`.

To train all trained from scratch baseline models for replicating results in the paper, do the following:
1. Train Standard on all datasets (MNIST, FMNIST, CIFAR10, SVHN, CIFAR100) for 5 random seeds each by running `./scripts/train_standard.sh`.
2. Train NC and OE on all datasets (MNIST, FMNIST, CIFAR10, SVHN, CIFAR100) with OOD dataset 300k random images for 5 random seeds each by running `./scripts/train_ood.sh tiny300k`.

## Training PreLoad:

To train a PreLoad model, run `python train_preload.py --dataset {MNIST, FMNIST, CIFAR10, SVHN, CIFAR100} --ood_data {tiny300k, imagenet, smooth, uniform} --lmbd {l} --lr {lr} --weight_decay {wd} --randseed {r}`. You can specify values for the arguments for lambda (l), learning rate (lr), and weight decay (wd).

To train all PreLoad models for replicating results in the paper, do the following:
1. Train PreLoad on all datasets with OOD dataset 300k random images for 5 random seeds each by running `./scripts/train_preload.sh tiny300k`.

## Training "Finetuned" Baselines:

To train a single "finetuned" OE-FT or Energy-FT baseline model, run `python train_ft.py --dataset {MNIST, FMNIST, CIFAR10, SVHN, CIFAR100} --method {oe, energy} --ood_data {tiny300k, imagenet, smooth, uniform} --m_in {in} --m_out {out} --randseed {r}`, where method can be OE-FT = `oe` or Energy-FT = `energy`. For Energy-FT, you can specify values for the arguments for in margin (in) and out margin (out). To train a single "finetuned" DOE-FT baseline model, run `python train_doe_ft.py --dataset {MNIST, FMNIST, CIFAR10, SVHN, CIFAR100} --ood_data {tiny300k, imagenet, smooth, uniform} --randseed {r} --epochs {e} --warmup {w}`. You can provide values for the arguments for total number of epochs (e) and number of warmup epochs (w). 

To train all finetuned baseline models for replicating results in the paper, do the following:
1. Make sure that Standard models have been trained before starting finetuning.
2. Train OE-FT and Energy-FT on all datasets with OOD dataset 300k random images for 5 random seeds each by running `./scripts/train_ft.sh tiny300k`.
3. Train DOE-FT on all datasets with OOD dataset 300k random images for 5 random seeds each by running `./scripts/train_doe_ft.sh tiny300k`.

## Training PreLoad-FT:

To train a PreLoad-FT model, first run `python train_posthoc.py --dataset {MNIST, FMNIST, CIFAR10, SVHN, CIFAR100} --ood_data {tiny300k, imagenet, smooth, uniform} --lmbd {l} --lr {lr} --weight_decay {wd} --randseed {r}` to initialize the added weights of the PreLoad network before they are finetuned with the rest of the PreLoad network. Then run `python train_preload_ft.py --dataset {MNIST, FMNIST, CIFAR10, SVHN, CIFAR100} --ood_data {tiny300k, imagenet, smooth, uniform} --lmbd {l} --lr {lr} --weight_decay {wd} --randseed {r}`. In both cases, you can specify values for the arguments for lambda, learning rate, and weight decay.

To train all PreLoad-FT models for replicating results in the paper, do the following:
1. Make sure that Standard models have been trained before starting finetuning.
2. Train the additional weights of the PreLoad network (weights of the extra class) on all datasets with OOD dataset 300k random images for 5 random seeds each by running `./scripts/train_posthoc.sh tiny300k`. Make sure the training is completed before starting the next step.
3. Fine-tune the full PreLoad network by training PreLoad-FT on all datasets with OOD dataset 300k random images for 5 random seeds each by running `./scripts/train_preload_ft.sh tiny300k`.

## Evaluating Models:

Run the evaluation script to evaluate the models for all methods and datsets: `./scripts/eval.sh`. Also, run `./scripts/eval_shift.sh` to perform evaluation for the dataset shift experiment. This will generate the results for the paper in the `results` directory.

## Reproducing the Paper's Tables:

To generate tables, use `aggregate.py`. Specify the type of metric you want in the table using the argument `--type`. The options are ('auroc', 'auprc', 'fpr95', 'mmc', 'acc', 'cal'). If you would like to get the results for auroc, aurprc, fpr95, or mmc averaged over all the OOD test sets (except for FarAway and FarAway-RD), add `--averaged True`.

For instance, Table 2 in the paper can be generated using: `python aggregate.py --type fpr95 --averaged True`.

## Reproducing the Paper's Figures:

To generate Figure 3 in the paper, run `python plot_MNIST-R.py` and `python plot_CIFAR10-C.py`.

## Hyperparameter Tuning:

We used Weights and Biases (wandb) for hyperparameter tuning. We did hyperparameter tuning for optimal in-domain margin and OOD margin for Energy-FT for all datasets, for optimal learning rate and weight decay for PreLoad for CIFAR10 and CIFAR100 datasets, and for optimal learning rate, weight decay, and lambda for PreLoad-FT for all datasets.

When you run the code using the steps above, the optimal hyperparameter values we found will be used by default, as long as you don't specify your own values for them.

## References

1. Code adapted from https://github.com/wiseodd/bayesian_ood_training.
2. Copied code from https://github.com/omegafragger/DDU/blob/main/utils/gmm_utils.py into `PreLoad/util/gmm_utils.py` for DDU.
3. Copied code from https://github.com/QizhouWang/DOE/blob/main/doe_final.py into `PreLoad/train_doe_ft.py` and from https://github.com/QizhouWang/DOE/blob/main/utils/utils_awp.py into `PreLoad/util/awp.py` for DOE-FT.

