#! /bin/bash

#Run this script in your local installation of chemprop (changing '--datapath' appropriately) with 'E-pi-pi', 'Z-pi-pi', 'E-n-pi', or 'Z-n-pi' as input arguments to obtain the DMPNN rmse performance. 
mkdir tmp
echo "Training DMPNN on "$1"...\n"
python hyperparameter_optimization.py --data_path ../datasets/chemprop/$1.can --dataset_type regression --num_iters 10 --config_save_path tmp/$1.json
for i in {1..20}; do
# change 'rmse' to 'r2' or 'mae' to obtain those performance metrics
python train.py --data_path ../datasets/chemprop/photoswitches_$1.can --config_path tmp/$1.json --dataset_type regression --split_sizes 0.7 0.1 0.2 --quiet --metric rmse --features_generator rdkit_2d_normalized --no_features_scaling --epochs 50 --seed $i
done

