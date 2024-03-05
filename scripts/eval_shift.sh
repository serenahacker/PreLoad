#!/bin/bash

declare -a dsets=("MNIST" "CIFAR10")

for dset in "${dsets[@]}";
do
    python eval_shift.py --dataset $dset --ood_data tiny300k
done

