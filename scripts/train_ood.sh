#!/bin/bash

if [ -z "$1" ] || { [[ "$1" != "tiny300k" ]] && [[ "$1" != "imagenet" ]] && [[ "$1" != "smooth" ]] && [[ "$1" != "uniform" ]]; }; then
    echo "Usage: ./train_ood.sh ood_data"
    echo "ood_data: tiny300k, imagenet, smooth, uniform"
    echo "Example: ./train_ood.sh tiny300k"
    exit 1
fi

declare -a dsets=("MNIST" "FMNIST" "SVHN" "CIFAR10" "CIFAR100")
declare -a ood_methods=("noneclass" "oe")

for dset in "${dsets[@]}";
do
    for ood_method in "${ood_methods[@]}";
    do
        for i in {1..5};
        do
            python train.py --dataset $dset --method $ood_method --ood_data $1 --randseed $i
        done
    done
done
