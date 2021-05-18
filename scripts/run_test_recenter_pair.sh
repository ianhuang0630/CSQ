#!/bin/bash


# this script runs the IOU and mesh-prediction evaluation for any model. It does so for NON-baseline models. For baseline models, look at 'run_test_baseline.sh'

./evaluate.py /orion/group/ShapeNetManifold_10000/03001627/ mydecomp2_IOU_recenter/ --n_primitives 20 --weight_file myruns/chair/28HR3REP1/model_149 --train_with_bernoulli --use_sq --use_chamfer --dataset_type shapenet_v2 --run_on_gpu  --train_test_splits_file ../Chair_train_val_test_split.csv --save_individual_IOUs --save_prediction_as_mesh --recenter_superquadrics
