"""
A preprocessing script that samples N points
"""

import sys
from multiprocessing import Pool
import numpy as np
import os
import trimesh
import pickle
from metrics import get_signed_distance
from tqdm import tqdm


np.seed(7)
N = 10000 # default value
part_output_dir = '/orion/u/ianhuang/superquadric_parsing/parts_output'
category_id = "03001627"
samples_output_dir = '/orion/u/ianhuang/superquadric_parsing/parts_randindex_samples'


def membership2neighborpairs(membership):
    """
    Membership is a list, where each element corresponds to a list of partid's that a single
    point corresponds to. There are N elements within membership

    """
    pairs = []
    for i in range(len(membership)):
        for j in range(i+1, len(membership)):
            # are points i and j related?
            if len(set(membership[i]).intersection(set(membership[j]))) > 0:
                pairs.append((i, j))
    pairs = np.array(pairs).transpose() # 2xnum_pairs
    return pairs


def sample_N_points_within_cvxs(N, leaves, use_bbox_as_backup=False):
    """
    Returns a list of N points that are within the meshes. (Nx3)
    For each point, a membership list is also returned.
    """

    # sample proportional to the bounding box size

    meshes = [leaf['convex_mesh'] for leaf in leaves
              if leaf['convex_mesh'] is not None]
    mins = np.min(np.vstack([np.min(mesh.vertices, axis=0)
                             for mesh in meshes]), axis=0)
    maxs = np.max(np.vstack([np.max(mesh.vertices, axis=0)
                             for mesh in meshes]), axis=0)

    # some of these meshes are not going to have a convex.
    # use_bbox_as_backup will step in and be used as a subsitute in that case
    # if `use_bbox_as_backup == True`.

    cvx_meshes = []
    part_ids = []
    for leaf in leaves:
        if leaf['convex_mesh'] is None:
            if use_bbox_as_backup:
                cvx_meshes.append(leaf['bbox_mesh'])
                part_ids.append(leaf['id'])
            else:
                continue
        else:
            cvx_meshes.append(leaf['convex_mesh'])
            part_ids.append(leaf['id'])

    # sample till all points fall into the desired convexes

    samples = [] # this will eventually be a Nx3 matrix
    memberships = [] # this will eventually be a length N_list, where each list is a list of part_ids
    while True:
        # sample N points
        candidates = np.vstack([np.random.uniform(mins[i], maxs[i], N)
                                for i in range(3)]).transpose()

        # parallel process the following:
        parts_proximityQueries = [(part_ids[i],
                                   trimesh.proximity.ProximityQuery(cvx_meshes[i]),
                                   candidates)
                                  for i in range(len(cvx_meshes))] # some parts are tiny

        # query which ones are inside
        with Pool(8) as p:
            dists_per_part = p.map(get_signed_distance,
                                   parts_proximityQueries)

        part_id_list = [el[0] for el in dists_per_part]

        # distance_matrices NxP
        distance_per_part = np.array([el[1] for el in dists_per_part]).transpose()

        membership = distance_per_part >= 0
        point_idxs, part_id_idxs = np.where(membership)
        membership_per_sample = [[]] * membership.shape[0]  # length N list

        for i in range(len(point_idxs)):
            membership_per_sample[point_idxs[i]]\
                .append(part_id_list[part_id_idxs[i]])

        for point_idx, membership in enumerate(membership_per_sample):
            if len(membership) > 0 and len(samples) < N:
                samples.append(candidates[point_idx])
                memberships.append(membership)
            if len(samples) == N:
                break

        # if we've found N of them, then we're done.
        if len(samples) == N:
            break

    samples = np.vstack(samples)

    return samples, memberships

if __name__ == '__main__':
    import ipdb; ipdb.set_trace()
    if len(sys.argv) > 1:
        N = int(sys.argv[1])
    if len(sys.argv) > 2:
        part_output_dir = sys.argv[2]
    if len(sys.argv) > 3:
        category_id = sys.argv[3]
    if len(sys.argv) > 4:
        samples_output_dir = sys.argv[4]

    assert os.path.exists(os.path.join(part_output_dir, category_id))

    # now, for each model, let's load their parts first.

    models = os.listdir(os.path.join(part_output_dir,  category_id))
    if not os.path.isdir(samples_output_dir):
        os.makedirs(samples_output_dir)

    for model in tqdm(models):
        with open(os.path.join(part_output_dir, category_id, model), 'rb') as f:
            leaves = pickle.load(f)
        part_cvxs = [leaf['convex_mesh'] for leaf in leaves]
        part_ids = [leaf['id'] for leaf in leaves]

        print('sampling for {}'.format(model[:-len('.pkl')]))
        samples, memberships = sample_N_points_within_cvxs(N, leaves,
                                                           use_bbox_as_backup = True)

        neighborpairs = membership2neighborpairs(memberships)

        print('Done.')

        # save the samples
        model_name = model[:-len('.pkl')]
        pickle_path = os.path.join(samples_output_dir, '{}.pkl'.format(model_name))

        # samples_output_dir/model_name.pkl
        with open(pickle_path, 'wb') as f:
            pickle.dump({'samples': samples,
                         'part_id_membership': memberships,
                         'neighbor_pairs': neighborpairs}, f)

        print('Saved at {}.'.format(pickle_path))





