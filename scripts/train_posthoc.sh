#!/bin/bash

if [ -z "$1" ] || { [[ "$1" != "tiny300k" ]] && [[ "$1" != "imagenet" ]] && [[ "$1" != "smooth" ]] && [[ "$1" != "uniform" ]]; }; then
    echo "Usage: ./train_posthoc.sh ood_data"
    echo "ood_data: tiny300k, imagenet, smooth, uniform"
    echo "Example: ./train_posthoc.sh tiny300k"
    exit 1
fi

declare -a dsets=("MNIST" "FMNIST" "SVHN" "CIFAR10" "CIFAR100")

for dset in "${dsets[@]}";
do
    for i in {1..5};
    do
        python train_posthoc.py --dataset $dset --ood_data $1 --randseed $i
    done
done
