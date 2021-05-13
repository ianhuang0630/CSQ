#!/bin/bash

# this script runs the baseline with the designated training testing splits.




# ./train_network.py /orion/group/ShapeNetManifold_10000/04256520/ myruns/ --use_sq --lr 1e-4 --n_primitives 20 --train_with_bernoulli --dataset_type shapenet_v2 --use_chamfer --run_on_gpu
# ./train_network.py /orion/group/ShapeNetManifold_10000/04379243/ myruns/table/ --use_sq --lr 1e-4 --n_primitives 20 --train_with_bernoulli --dataset_type shapenet_v2 --use_chamfer --run_on_gpu
# ./train_network.py /orion/group/ShapeNetManifold_10000/03001627/ myruns/chair/ --use_sq --lr 1e-4 --n_primitives 20 --train_with_bernoulli --dataset_type shapenet_v2 --use_chamfer --run_on_gpu
# ./train_network.py /orion/group/ShapeNetManifold_10000/02958343/ myruns/car/ --use_sq --lr 1e-4 --n_primitives 20 --train_with_bernoulli --dataset_type shapenet_v2 --use_chamfer --run_on_gpu
# ./train_network.py /orion/group/ShapeNetManifold_10000/02691156/ myruns/airplane/ --use_sq --lr 1e-4 --n_primitives 20 --train_with_bernoulli --dataset_type shapenet_v2 --use_chamfer --run_on_gpu --batch_size 16
# ./train_network.py /orion/group/ShapeNetManifold_10000/02828884/ myruns/bench/ --use_sq --lr 1e-4 --n_primitives 20 --train_with_bernoulli --dataset_type shapenet_v2 --use_chamfer --run_on_gpu
# ./train_network.py /orion/group/ShapeNetManifold_10000/02871439/ myruns/bookshelf/ --use_sq --lr 1e-4 --n_primitives 20 --train_with_bernoulli --dataset_type shapenet_v2 --use_chamfer --run_on_gpu
# ./train_network.py /orion/group/ShapeNetManifold_10000/02933112/ myruns/cabinet/ --use_sq --lr 1e-4 --n_primitives 20 --train_with_bernoulli --dataset_type shapenet_v2 --use_chamfer --run_on_gpu


./train_network.py /orion/group/ShapeNetManifold_10000/03001627/ myruns/chair_traintest/ --use_sq --lr 1e-4 --n_primitives 20 --train_with_bernoulli --dataset_type shapenet_v2 --use_chamfer --run_on_gpu --train_test_splits_file ../Chair_train_val_test_split.csv
