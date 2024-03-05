#!/bin/bash

declare -a dsets=("MNIST" "FMNIST" "SVHN" "CIFAR10" "CIFAR100")

for dset in "${dsets[@]}";
do
    python eval.py --dataset $dset --ood_data tiny300k
done

