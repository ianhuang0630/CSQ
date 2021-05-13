#!/bin/bash

# Running test for baseline model

./evaluate.py /orion/group/ShapeNetManifold_10000/03001627/ mydecomp2_IOU_baseline/ --n_primitives 20 --weight_file myruns/chair/4NVH1YPUM/model_149 --train_with_bernoulli --use_sq --use_chamfer --dataset_type shapenet_v2 --run_on_gpu  --train_test_splits_file ../Chair_train_val_test_split.csv --save_individual_IOUs --save_prediction_as_mesh
