import scipy as sp
import numpy as np
import trimesh
from tqdm import tqdm
from multiprocessing import Pool

RANDOM_SAMPLING = True

def get_signed_distance(input_):
    id_ = input_[0]
    query = input_[1]
    samples = input_[2]
    signed_distances = query.signed_distance(samples)
    return (id_, signed_distances)

def sample_from_bbox(ranges, num_samples):
    samples = np.vstack([np.random.uniform(ranges['min'][i],
                                           ranges['max'][i],
                                           num_samples)
                         for i in range(3)]).transpose()
    return samples


def update_IOUs(matrix, superquadrics, part_bboxes, part_meshes, partid2idx):
    """
    Args:
        matrix (np.ndarray): |superquadrics|x|part_meshes|
        superquadrics (list): every element is [superquadric_id, superquadric_mesh]
        part_meshes (list of bboxes with id's): every element is [id, bbox_mesh]
        partid2idx (dict): dictionary mapping from part id's to column indices in matrix
    Returns:
        matrix
    """

    
    # IOU = # pts in both/ # pts in either.
    id_bbox_supe = []
    for id_, superquadric_mesh in superquadrics:
        id_bbox_supe.append((id_, {'min': np.min(superquadric_mesh.vertices, axis=0),
                                   'max': np.max(superquadric_mesh.vertices, axis=0)}))
    id_bbox_part= []
    for id_, part_mesh in part_meshes:
        id_bbox_part.append((id_, {'min': np.min(part_mesh.vertices, axis=0),
                             'max': np.max(part_mesh.vertices, axis=0)}))

    # for each bounding box, sample some points based on the volume of the box

    density = 1e4
    id_samples_supe = []
    id_samples_part = []
    for id_, ranges in id_bbox_supe:
        volume = np.prod(ranges['max'] - ranges['min'])
        num_samples = int(np.round(volume*density))
        samples = sample_from_bbox(ranges, num_samples)
        id_samples_supe.append((id_, samples))

    # part 18 vol: volume 3.744644592834477e-05 , # samples: 0
    # part 17 vol: volume 3.08617280773189e-05 , # samples: 0
    # part 16 vol: volume 0.0026231809821468223 , # samples: 26
    # part 15 vol: volume 0.0026886881491884275 , # samples: 27
    # part 14 vol: volume 0.001284182319198027 , # samples: 13
    # part 13 vol: volume 0.0012846363234832224 , # samples: 13
    # part 10 vol: volume 2.2033962642730202e-05 , # samples: 0
    # part 9 vol: volume 2.1039300923017336e-05 , # samples: 0
    # part 8 vol: volume 1.1773841260435058e-05 , # samples: 0
    # part 7 vol: volume 2.5730767925438312e-05 , # samples: 0
    # part 6 vol: volume 0.016359085413370177 , # samples: 164
    # part 3 vol: volume 0.02414892319913179 , # samples: 241


    for id_, ranges in id_bbox_part:
        volume = np.prod(ranges['max'] - ranges['min'])
        num_samples = int(np.round(volume*density))
        samples = sample_from_bbox(ranges, num_samples)
        id_samples_part.append((id_, samples))


    # question: for each set of samples, how many belong in their correspodning meshes?
    superquadrics_proximityQueries = [(superquadrics[i][0],
                                       trimesh.proximity.ProximityQuery(superquadrics[i][1]),
                                       id_samples_supe[i][1])
                                      for i in range(len(superquadrics)) if
                                      len(id_samples_supe[i][1]) > 0]


    parts_proximityQueries = [(part_meshes[i][0],
                               trimesh.proximity.ProximityQuery(part_meshes[i][1]),
                               id_samples_part[i][1])
                              for i in range(len(part_meshes)) if
                              len(id_samples_part[i][1]) > 0] # some parts are tiny

    print('calculating membership for superquadrics')
    with Pool(8) as p:
        dists_per_supe = p.map(get_signed_distance,
                               superquadrics_proximityQueries)

    print('calculating membership for part')
    with Pool(8) as p:
        dists_per_part = p.map(get_signed_distance,
                               parts_proximityQueries)

    delta = np.zeros_like(matrix)
    intersections = {}
    supe_in = {}
    part_in = {}
    # for supe_id, s_ in superquadrics:
    #     supe_in[supe_id] = 0
    #     for part_id, p_ in part_meshes:
    #         intersections[(supe_id, part_id)] = 0 

    # for part_id, p_ in part_meshes:
    #     part_in[part_id] = 0

    for supe_id, supe_dist in dists_per_supe:
        num_inside = np.sum(supe_dist >= 0)
        supe_in[supe_id] = num_inside

    for part_id, part_dist in dists_per_part:
        num_inside = np.sum(part_dist >= 0)
        part_in[part_id] = num_inside

    # for supe_id, supe_pq in tqdm(superquadrics_proximityQueries):
    #     supe_dist = supe_pq.signed_distance(samples)
    #     num_inside = np.sum(supe_dist > 0)
    #     supe_in[supe_id] += num_inside
    #     dists_per_supe.append((supe_id, supe_dist))

    # for part_id, part_pq in tqdm(part_proximityQueries):
    #     part_dist = part_pq.signed_distance(samples)
    #     num_inside = np.sum(part_dist > 0)
    #     part_in[part_id] += num_inside
    #     dists_per_part.append((part_id, part_dist))

    print ('calculating intersections')
    for supe_id, supe_mesh_query, supe_samples in superquadrics_proximityQueries:
        # for this supe, how does t overlap with the parts?
        overlap_queries = [(part_id, supe_mesh_query, part_samples)
                           for part_id, _, part_samples
                           in parts_proximityQueries]
        with Pool(8) as p:
            dists_per_part = p.map(get_signed_distance,
                                   overlap_queries)

        for part_id, part_dist in dists_per_part:
            num_inside_intersection = np.sum(part_dist >= 0)
            intersections[(supe_id, part_id)] = num_inside_intersection

            # calculating iou
            delta[supe_id, partid2idx[part_id]] = \
                intersections[(supe_id, part_id)]\
                /(supe_in[supe_id] + part_in[part_id] -
                  intersections[(supe_id, part_id)])

    # matrix = matrix + delta
    matrix = matrix+delta

    return matrix, delta

def get_consistency(matrix):
    # matrix should be rectangular 

    super_quad_idx, part_idx = \
        sp.optimize.linear_sum_assignment(matrix,
                                      maximize=True)

    best_sum_IOU = matrix[super_quad_idx, part_idx].sum()
    superquadidx2partid = zip(super_quad_idx.tolist(), part_idx.tolist())

    return best_sum_IOU, superquadidx2partid


