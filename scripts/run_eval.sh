#!/bin/bash

# This runs the model on shapenet inputs, and saves their meshes.
# it does a subset of what run_test*.sh can do.

# ./forward_pass.py /orion/group/ShapeNetManifold_10000/04256520/ mydecomp/ --model_tag a87440fc1fe87c0d6bd4dcee82f7948d --n_primitives 20 --weight_file myruns/sofa/13NG0DGQZ/model_101 --train_with_bernoulli --use_sq --use_chamfer --dataset_type shapenet_v2 --run_on_gpu --save_prediction_as_mesh

# ./forward_pass.py /orion/group/ShapeNetManifold_10000/04256520/ mydecomp2/ --n_primitives 20 --weight_file myruns/sofa/13NG0DGQZ/model_149 --train_with_bernoulli --use_sq --use_chamfer --dataset_type shapenet_v2 --run_on_gpu --save_prediction_as_mesh
# ./forward_pass.py /orion/group/ShapeNetManifold_10000/03001627/ mydecomp2/ --n_primitives 20 --weight_file myruns/chair/4NVH1YPUM/model_149 --train_with_bernoulli --use_sq --use_chamfer --dataset_type shapenet_v2 --run_on_gpu --save_prediction_as_mesh
# ./forward_pass.py /orion/group/ShapeNetManifold_10000/02958343/ mydecomp2/ --n_primitives 20 --weight_file myruns/car/ENWQMYXAP/model_149 --train_with_bernoulli --use_sq --use_chamfer --dataset_type shapenet_v2 --run_on_gpu --save_prediction_as_mesh
# ./forward_pass.py /orion/group/ShapeNetManifold_10000/04379243/ mydecomp2/ --n_primitives 20 --weight_file myruns/table/FTPDRCSNQ/model_149 --train_with_bernoulli --use_sq --use_chamfer --dataset_type shapenet_v2 --run_on_gpu --save_prediction_as_mesh
# ./forward_pass.py /orion/group/ShapeNetManifold_10000/02828884/ mydecomp2/ --n_primitives 20 --weight_file myruns/bench/MI30KAG5C/model_149 --train_with_bernoulli --use_sq --use_chamfer --dataset_type shapenet_v2 --run_on_gpu --save_prediction_as_mesh
# ./forward_pass.py /orion/group/ShapeNetManifold_10000/02871439/ mydecomp2/ --n_primitives 20 --weight_file myruns/bookshelf/JZ58DEAJZ/model_149 --train_with_bernoulli --use_sq --use_chamfer --dataset_type shapenet_v2 --run_on_gpu --save_prediction_as_mesh
# ./forward_pass.py /orion/group/ShapeNetManifold_10000/02933112/ mydecomp2/ --n_primitives 20 --weight_file myruns/cabinet/MDPW99S03/model_149 --train_with_bernoulli --use_sq --use_chamfer --dataset_type shapenet_v2 --run_on_gpu --save_prediction_as_mesh


# still waiting on: NEED MODEL DIR
# ./forward_pass.py /orion/group/ShapeNetManifold_10000/02691156/ mydecomp2/ --n_primitives 20 --weight_file myruns/airplane/9CQRY4C18/model_149 --train_with_bernoulli --use_sq --use_chamfer --dataset_type shapenet_v2 --run_on_gpu --save_prediction_as_mesh

# for pairs
./forward_pass.py /orion/group/ShapeNetManifold_10000/03001627/ mydecomp2_pair/ --n_primitives 20 --weight_file myruns/chair/28HR3REP1/model_149 --train_with_bernoulli --use_sq --use_chamfer --dataset_type shapenet_v2 --run_on_gpu --save_prediction_as_mesh
