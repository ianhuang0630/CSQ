"""
A preprocessing script that samples N points within the bounding box of a shape
"""
import sys
import numpy as np
import os
from compute_transmat_shapenetv1_to_partnet import load_obj
from tqdm import tqdm

N = 10000
original_meshes_dir =  '/orion/group/ShapeNetManifold_10000'
category_id = "03001627"
output_dir = "/orion/u/ianhuang/superquadric_parsing/shape_uniform_bbox_samples"

if __name__ == '__main__':
    if len(sys.argv) > 1 :
        N = int(sys.argv[1])
    if len(sys.argv) > 2 :
        original_meshes_dir = sys.argv[2]
    if len(sys.argv) > 3 :
        category_id = sys.argv[3]
    if len(sys.argv) > 4 :
        output_dir = sys.argv[4]

    # load the model's mesh, then get
    model_dir = os.path.join(original_meshes_dir, category_id)
    assert os.path.isdir(model_dir)

    models = os.listdir(model_dir)

    if not os.path.isdir(output_dir):
        os.makedirs(output_dir)

    for model in tqdm(models):

        obj_path = os.path.join(original_meshes_dir,
                                model, 'model_normalized.obj')
        curr_vs, curr_fs = load_obj(obj_path)

        mins = np.min(curr_vs, axis=0)
        maxs = np.max(curr_vs, axis=0)

        samples = np.vstack([np.random.uniform(mins[i], maxs[i], N)
                             for i in range(3)]).transpose()

        # save samples
        np_output_path = os.path.join(output_dir, '{}.npy'.format(model))
        np.save(samples, np_output_path)








    pass

