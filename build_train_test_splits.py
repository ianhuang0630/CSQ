"""
Creates the training testing split used in train_network.py
"""

import os
from get_parts import *
import numpy as np
import random
import pandas as pd

random.seed(40)


# TODO:create csv. names=["id", "synsetId", "subSynsetId", "modelId", "split"]
# split -> "train", "test", "val"
# modelId -> model_id
# id -> ?
TRAINING_FRAC = 0.6
VALIDATION_FRAC = 0.2
TESTING_FRAC = 0.2

# list all the sid's available under /Manifold_10000/OBJ_CAT_ID

STRUCTURENET_DIR = "partnethiergeo"
PARTNET_DIR = "/orion/group/PartNet/data_v0"
SHAPENET_DIR = "/orion/group/ShapeNetManifold_10000/"
OBJ_CAT = "Chair"
OBJ_CAT_ID = "03001627"

shapenet_sid_list = os.listdir(os.path.join(SHAPENET_DIR, OBJ_CAT_ID))
shapenet_sid_set = set(shapenet_sid_list)


pid_list  = os.listdir(PARTNET_DIR) # all of the pid's available
res = collect_partnet_shapenet_correspondences(os.path.dirname(PARTNET_DIR))
cat2pid2sid = res['pid2sid']
cat2sid2pid = res['sid2pid']

partnet_sid_set = set([])
for pid_key in cat2pid2sid[OBJ_CAT]:
    partnet_sid_set.add(cat2pid2sid[OBJ_CAT][pid_key][0])
    # NOTE: this can be done because the map is many-to-one


# what's the intersection between the two sid's
have_parts = partnet_sid_set.intersection(shapenet_sid_set)
have_no_parts = shapenet_sid_set - have_parts
print("{}/{} inside the shapenet dataset for category {} have parts".format(len(have_parts),
                                                                            len(shapenet_sid_set),
                                                                            OBJ_CAT))
print("{}/{} inside the partnet dataset for category {} are in shapenet".format(len(have_parts),
                                                                            len(partnet_sid_set),
                                                                            OBJ_CAT))

have_parts = list(have_parts)
have_no_parts = list(have_no_parts)
random.shuffle(have_parts)
random.shuffle(have_no_parts)

modelId = have_parts + have_no_parts
to_testing = int(np.round(TESTING_FRAC*len(modelId)))
to_validation = int(np.round(VALIDATION_FRAC * len(modelId)))
to_train = int(np.round(TRAINING_FRAC * len(modelId)))

split = ['test']*to_testing + ['val']*to_validation + ['train']*to_train

# other fields that don't matter:
# subSynsetId
# synsetId
# id
data = {'id': [np.nan]*len(split),
        'synsetId': [np.nan]*len(split),
        'subSynsetId': [np.nan]*len(split),
        'modelId': modelId,
        'split': split
        }

train_test_split_df = pd.DataFrame.from_dict(data, orient='columns')
train_test_split_df.to_csv('{}_train_val_test_split.csv'.format(OBJ_CAT))

print("Done.")
# structurenet_json = os.path.join(STRUCTURENET_DIR,
#     '{}_hier'.format(OBJ_CAT), '{}.json'.format(model_id))
# assert(os.path.exists(structurenet_json))
# partnet_json = os.path.join(PARTNET_DIR, str(model_id),
#         'result_after_merging.json')
# assert(os.path.exists(partnet_json))
# leaves = collect_leaf_nodes(structurenet_json)
# leaves = find_corresponding_meshes(partnet_json, leaves)
# partnet_mesh_dir = os.path.join(PARTNET_DIR, str(model_id), 'objs')
# leaves = load_meshes(leaves, partnet_mesh_dir)
