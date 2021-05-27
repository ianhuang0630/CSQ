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


np.random.seed(7)

N = 1000 # default value
part_output_dir = '/orion/u/ianhuang/superquadric_parsing/parts_output'
category_id = "03001627"
samples_output_dir = '/orion/u/ianhuang/superquadric_parsing/parts_randindex_samples'


def membership2neighborpairs(membership):
    """
    Membership is a list, where each element corresponds to a list of partid's that a single
    point corresponds to. There are N elements within membership

    """

    # faster way:
    groups2pt_idx = {}
    for pt_idx, groups in enumerate(membership):
        for group in groups:
            if group not in groups2pt_idx:
                groups2pt_idx[group] = [pt_idx]
            else:
                groups2pt_idx[group].append(pt_idx)

    pairs = set([])
    for group in groups2pt_idx:
        pt_indices = groups2pt_idx[group]
        for i in pt_indices:
            for j in pt_indices:
                pairs.add((i , j))

    pairs = list(pairs)

    # pairs = []
    # for i in range(len(membership)):
    #     for j in range(i+1, len(membership)):
    #         # are points i and j related?
    #         if len(set(membership[i]).intersection(set(membership[j]))) > 0:
    #             pairs.append((i, j))
    pairs = np.array(pairs).transpose() # 2xnum_pairs
    return pairs

def get_bbox_volume(leaf):

    return (leaf['bbox']['x_max'] - leaf['bbox']['x_min'])\
        * (leaf['bbox']['y_max'] - leaf['bbox']['y_min'])\
        * (leaf['bbox']['z_max'] - leaf['bbox']['z_min'])

def generate_N_points_within_bbox(N, bbox_bounds):

    mins = np.array([bbox_bounds['x_min'], bbox_bounds['y_min'], bbox_bounds['z_min']])
    maxs = np.array([bbox_bounds['x_max'], bbox_bounds['y_max'], bbox_bounds['z_max']])

    return np.vstack([np.random.uniform(mins[i], maxs[i], N)
                      for i in range(3)]).transpose()

def sample_N_points_within_cvxs(N, leaves, use_bbox_as_backup=False):
    """
    Returns a list of N points that are within the meshes. (Nx3)
    For each point, a membership list is also returned.
    """

    # some of these meshes are not going to have a convex.
    # use_bbox_as_backup will step in and be used as a subsitute in that case
    # if `use_bbox_as_backup == True`.

    cvx_meshes = []
    part_ids = []
    volumes = []
    bboxes = []
    for leaf in leaves:
        if leaf['convex_mesh'] is None:
            if use_bbox_as_backup:
                cvx_meshes.append(leaf['bbox_mesh'])
            else:
                continue
        else:
            cvx_meshes.append(leaf['convex_mesh'])

        part_ids.append(leaf['id'])
        volumes.append(get_bbox_volume(leaf))
        bboxes.append(leaf['bbox'])

    # sample till all points fall into the desired convexes
    total_volume = sum(volumes)
    samples_per_leaf_bbox = [int(volumes[i]/total_volume * N) for i in range(len(volumes)-1)]
    samples_per_leaf_bbox += [N - sum(samples_per_leaf_bbox)]

    assert len(samples_per_leaf_bbox) == len(volumes)

    samples = [] # this will eventually be a Nx3 matrix
    memberships = [] # this will eventually be a length N_list, where each list is a list of part_ids

    while True:
        # sample N points
        candidates = np.vstack([generate_N_points_within_bbox(samples_per_leaf_bbox[i],
                                                    bboxes[i])
                                for i in range(len(volumes))])
        np.random.shuffle(candidates)

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
        membership_per_sample = [[] for i in range(membership.shape[0])]  # length N list

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





