#!bin/bash


DEVICES=0

MODEL=ResNet18_flanc
CUDA_VISIBLE_DEVICES=$DEVICES python main_FL.py --n_agents 100 --dir_data ../ --data_train cifar10  --n_joined 10 --split iid --local_epochs 2 --batch_size 32 --epochs 500 --decay step-250-375 --lr 0.1 --fraction_list 0.25,0.5,0.75,1 --project FLANC_CIFAR10 --template ResNet18 --model ${MODEL} --basis_fraction 0.125 --n_basis 0.25 --save FLANC  --dir_save ../experiment --save_models

MODEL=cnn_flanc
CUDA_VISIBLE_DEVICES=$DEVICES python main_FL.py --n_agents 100 --dir_data ../ --data_train fashion-mnist  --n_joined 10 --split iid --local_epochs 1 --batch_size 32 --epochs 200 --decay step-100 --lr 0.01 --fraction_list 0.25,0.5,0.75,1 --project FLANC --template ResNet18 --model ${MODEL} --basis_fraction 0.125 --n_basis 0.25 --save FLANC  --dir_save ../experiment --save_models





