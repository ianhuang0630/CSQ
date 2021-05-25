"""
A preprocessing script that samples N points
"""

import sys
import numpy as np
import os
import trimesh
import pickle

np.seed(7)
N = 10000 # default value
part_output_dir = '/orion/u/ianhuang/superquadric_parsing/parts_output'
category_id = "03001627"

def sample_N_points_within(N, meshes):
    """
    Returns a list of N points that are within the meshes. (Nx3)
    For each point, a membership list is also returned.
    """

    # sample proportional to the bounding box size

    mins = np.min(np.vstack([np.min(mesh.vertices, axis=0) for mesh in meshes]), axis=0)
    maxs = np.max(np.vstack([np.max(mesh.vertices, axis=0) for mesh in meshes]), axis=0)

    while True:
        # mins


    pass

if __name__ == '__main__':
    import ipdb; ipdb.set_trace()
    if len(sys.argv) > 1:
        N = int(sys.argv[1])
    if len(sys.argv) > 2:
        part_output_dir = sys.argv[2]
    if len(sys.argv) > 3:
        category_id = sys.argv[3]

    assert os.path.exists(os.path.join(part_output_dir, category_id))

    # now, for each model, let's load their parts first.

    models = os.listdir(os.path.join(part_output_dir,  category_id))
    for model in models:
        with open(os.path.join(part_output_dir, category_id, model), 'rb') as f:
            leaves = pickle.load(f)
        part_cvxs = [leaf['convex_mesh'] for leaf in leaves]
        part_ids = [leaf['id'] for leaf in leaves]

        samples, memberships = sample_N_points_within(N, part_cvxs)


