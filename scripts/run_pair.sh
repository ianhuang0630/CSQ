#!/bin/bash

# this script trains the network given pairs.


# ./train_network.py /orion/group/ShapeNetManifold_10000/04256520/ myruns/ --use_sq --lr 1e-4 --n_primitives 20 --train_with_bernoulli --dataset_type shapenet_v2 --use_chamfer --run_on_gpu
# ./train_network.py /orion/group/ShapeNetManifold_10000/04379243/ myruns/table/ --use_sq --lr 1e-4 --n_primitives 20 --train_with_bernoulli --dataset_type shapenet_v2 --use_chamfer --run_on_gpu
# ./train_network.py /orion/group/ShapeNetManifold_10000/03001627/ myruns/chair/ --use_sq --lr 1e-4 --n_primitives 20 --train_with_bernoulli --dataset_type shapenet_v2 --use_chamfer --run_on_gpu
# ./train_network.py /orion/group/ShapeNetManifold_10000/02958343/ myruns/car/ --use_sq --lr 1e-4 --n_primitives 20 --train_with_bernoulli --dataset_type shapenet_v2 --use_chamfer --run_on_gpu
# ./train_network.py /orion/group/ShapeNetManifold_10000/02691156/ myruns/airplane/ --use_sq --lr 1e-4 --n_primitives 20 --train_with_bernoulli --dataset_type shapenet_v2 --use_chamfer --run_on_gpu --batch_size 16
# ./train_network.py /orion/group/ShapeNetManifold_10000/02828884/ myruns/bench/ --use_sq --lr 1e-4 --n_primitives 20 --train_with_bernoulli --dataset_type shapenet_v2 --use_chamfer --run_on_gpu
# ./train_network.py /orion/group/ShapeNetManifold_10000/02871439/ myruns/bookshelf/ --use_sq --lr 1e-4 --n_primitives 20 --train_with_bernoulli --dataset_type shapenet_v2 --use_chamfer --run_on_gpu
# ./train_network.py /orion/group/ShapeNetManifold_10000/02933112/ myruns/cabinet/ --use_sq --lr 1e-4 --n_primitives 20 --train_with_bernoulli --dataset_type shapenet_v2 --use_chamfer --run_on_gpu

./train_network_pair.py /orion/group/ShapeNetManifold_10000/03001627/ myruns2/chair_pairs/ --use_sq --lr 1e-4 --n_primitives 20 --train_with_bernoulli --dataset_type shapenet_v2 --use_chamfer --run_on_gpu --parsimony_regularizer_weight 0.0001 --overlapping_regularizer_weight 0.0001 --regularizer_type parsimony_regularizer 
