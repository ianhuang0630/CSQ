import json
import os
import trimesh
import numpy as np
from trimesh.exchange.export import export_mesh
import pickle

from compute_transmat_shapenetv1_to_partnet import get_shapenet2partnet_transformation_matrix

from tqdm import tqdm


def load_obj(fn):
    fin = open(fn, 'r')
    lines = [line.rstrip() for line in fin]
    fin.close()

    vertices = []; faces = [];
    for line in lines:
        if line.startswith('v '):
            vertices.append(np.float32(line.split()[1:4]))
        elif line.startswith('f '):
            faces.append(np.int32([item.split('/')[0] for item in line.split()[1:4]]))

    return np.vstack(vertices), np.vstack(faces)


def collect_partnet_shapenet_correspondences(partnet_dir):
    cat2pid2sid = {}
    cat2sid2pid = {}
    with open(os.path.join(partnet_dir, 'stats', 'all_valid_anno_info.txt'), 'r') as f:
        foo = f.readlines()
        foo = [el.strip().split(" ") for el in foo]

        for row in foo:
            partnet_id = row[0]
            category = row[2]
            shapenet_id = row[3]
            # things that I'm not as sure about.
            number = row[1]
            authors = row[4]

            if category not in cat2pid2sid:
                cat2pid2sid[category] = {}


            # this is one to one
            if partnet_id in cat2pid2sid[category]:
                cat2pid2sid[category][partnet_id].append(shapenet_id)
            else:
                cat2pid2sid[category][partnet_id] = [shapenet_id]

            # this is one to many, possibly. a single shapenet id can have multiple annotations.
            if category not in cat2sid2pid:
                cat2sid2pid[category] = {}
            if shapenet_id in cat2sid2pid[category]:
                tup = (partnet_id, number, authors)
                cat2sid2pid[category][shapenet_id].append(tup)
                # print('###')
                # for el in cat2sid2pid[category][shapenet_id]:
                #     print(el)

            else:
                tup = (partnet_id, number, authors)
                cat2sid2pid[category][shapenet_id] = [tup]

    return {'pid2sid': cat2pid2sid, 'sid2pid': cat2sid2pid}

def merge_meshes(V_list, F_list, labels=None):
    V = np.empty((0, 3), dtype=int)
    F = np.empty((0, 3), dtype=int)
    n_meshes = len(V_list)
    assert(len(F_list) == n_meshes)
    if labels is not None:
        assert(len(labels) == n_meshes)
        VL = np.empty(0, dtype=int)
        FL = np.empty(0, dtype=int)
    for i in range(n_meshes):
        nV = np.shape(V)[0]
        V = np.vstack((V, V_list[i]))
        F = np.vstack((F, (nV + F_list[i])))
        if labels is not None:
            n_vertices = np.shape(V_list[i])[0]
            n_faces = np.shape(F_list[i])[0]
            VL = np.concatenate((VL, np.array([labels[i]] * n_vertices)))
            FL = np.concatenate((FL, np.array([labels[i]] * n_faces)))
    if labels is not None:
        return V, F, VL, FL
    return V, F

def collect_leaf_nodes(structurenet_json):
    with open(structurenet_json, 'r') as f:
        data = json.load(f)
    data["level"] = 0
    queue = [data]
    leaves = []
    while queue:
        node = queue.pop()
        if 'children' not in node:
            leaves.append(node)
            # print(node.keys())
        else:
            for child in node['children']:
                child["level"] = node["level"] + 1
                child["label"] = node["label"] + "/" + child["label"]
                queue.append(child)
    return leaves

def find_corresponding_meshes(partnet_json, nodes):
    with open(partnet_json, 'r') as f:
        data = json.load(f)
    queue = data
    while queue:
        elem = queue.pop()
        for node in nodes:
            if node['id'] == elem['id']:
                node['objs'] = elem['objs']
        if 'children' in elem:
            for child in elem['children']:
                queue.append(child)
    return nodes

def load_meshes(nodes, in_dir):
    # TODO: it's possible that some of these nodes do not have valid obj's
    for node in nodes:
        in_files = [os.path.join(in_dir, '{}.obj'.format(x)) \
                for x in node['objs']] # a single part can have multiple obj's
        meshes = [trimesh.load(x) for x in in_files]
        V_list = [mesh.vertices for mesh in meshes]
        F_list = [mesh.faces for mesh in meshes]
        node['vertices'], node['faces'] = merge_meshes(V_list, F_list)
    return nodes


if __name__=='__main__':

    import pandas as pd
    import os

    STRUCTURENET_DIR = "/orion/u/ianhuang/superquadric_parsing/partnethiergeo"
    PARTNET_DIR = "/orion/group/PartNet/data_v0"
    # SHAPENET_DIR = "/orion/group/ShapeNetManifold_10000/"
    SHAPENET_DIR = "/orion/group/ShapeNetCore.v2/"
    OBJ_CAT = "Chair"
    OBJ_CAT_ID = "03001627"
    output_dir = '/orion/u/ianhuang/superquadric_parsing/parts_output'

    train_val_test_split = pd.read_csv(
        '../{}_train_val_test_split.csv'.format(OBJ_CAT),
        names=["id", "synsetId", "subSynsetId", "modelId", "split"]
    )

    test_indices = train_val_test_split['split'] == 'test'

    sid_list = train_val_test_split[test_indices].modelId.values.tolist()

    res = collect_partnet_shapenet_correspondences(os.path.dirname(PARTNET_DIR))
    res = res['sid2pid'][OBJ_CAT]

    # sid_list = ['1006be65e7bc937e9141f9b58470d646']
    # sid_list = ['69e6f0a5e903cda466ab323d8f805a57']

    for sid in tqdm(sid_list):
        pid = res[sid][0][0]
        # is this reallythe best  pid?
        structurenet_json = os.path.join(STRUCTURENET_DIR,
                                         '{}_hier'.format(OBJ_CAT.lower()),
                                         '{}.json'.format(pid))
        assert(os.path.exists(structurenet_json))
        partnet_json = os.path.join(PARTNET_DIR, str(pid),
                                    'result_after_merging.json')
        assert(os.path.exists(partnet_json))
        leaves = collect_leaf_nodes(structurenet_json)
        # partnet leaves: these are the sctual parts
        leaves = find_corresponding_meshes(partnet_json, leaves)

        number_leaves_w_obj = len([el for el in leaves if 'objs' in el])

        if number_leaves_w_obj == 0:
            print('{} doesnt have objs to load'.format(pid))
            continue
        if number_leaves_w_obj < len(leaves):
            no_obj_ids = [el['id'] for el in leaves if 'objs' not in el]
            for no_obj_id in no_obj_ids:
                print('partnet Id: {} Part id : {} is missing an obj'.format(pid, no_obj_id))
        # filter to only include the leaves that have it
        leaves = [leaf for leaf in leaves if 'objs' in leaf]

        partnet_mesh_dir = os.path.join(PARTNET_DIR, str(pid), 'objs')
        leaves = load_meshes(leaves, partnet_mesh_dir)
        # TODO: just print out the location of each of these meshes.
        # TODO: and their corresponding ID's and semantic labels

        rotation = np.array([[-1, 0, 0],
                             [0, 1, 0],
                             [0, 0, -1]])

        # transformation
        partnet_vertices = []
        ###### KAICHUN's method ####
        # transform = get_shapenet2partnet_transformation_matrix(pid) # np.load('output_transmat_shapenetv1_to_partnet/{}.npy'.format(pid))
        # rotation = np.array([[0, 0, 1],
        #                      [0, 1, 0],
        #                      [-1, 0, 0]])
        # for leaf in leaves:
        #     part_vertices = np.concatenate([leaf['vertices'], np.ones((leaf['vertices'].shape[0], 1), dtype=np.float32)], axis=1)
        #     part_vertices = part_vertices @ (np.linalg.inv(transform).T)
        #     part_vertices = part_vertices[:, :3]
        #     part_vertices = part_vertices @ (rotation.T)
        #     leaf['vertices'] = part_vertices
        ########

        for leaf in leaves:
            # make a mesh
            # leaf['id'] gives the instance id
            # leaf['label'] gives the semantics of what the part is
                        # leaf['vertices'] and leaf['faces'] can be used to construct obj's
            part_vertices = leaf['vertices']
            # part_vertices = np.concatenate([part_vertices, np.ones((part_vertices.shape[0], 1))], axis=1)
            # part_vertices = part_vertices @ (trans.T)
            part_vertices = part_vertices @ (rotation.T)
            # part_vertices = part_vertices[:, :3]

            leaf['vertices'] = part_vertices
            partnet_vertices.append(part_vertices)

        partnet_vertices = np.vstack(partnet_vertices)

        # now we rescale everything to the shapenet scale.
        # find the mesh of the whole object in manifold_10000

        shapenet_mesh_vertices, _ = load_obj(os.path.join(SHAPENET_DIR,
                                                          OBJ_CAT_ID,
                                                          sid,
                                                          'models',
                                                          'model_normalized.obj'))

        shapenet_max = np.max(shapenet_mesh_vertices, axis=0)
        shapenet_min = np.min(shapenet_mesh_vertices, axis=0)
        shapenet_center = 0.5*(shapenet_max + shapenet_min)
        shapenet_extent = shapenet_max - shapenet_min

        partnet_max = np.max(partnet_vertices, axis=0)
        partnet_min = np.min(partnet_vertices, axis=0)
        partnet_center = 0.5*(partnet_max + partnet_min)
        partnet_extent = partnet_max - partnet_min

        for leaf in leaves:
            leaf['vertices'] = leaf['vertices'] - partnet_center

        # shapenet_mesh_max = np.max(np.linalg.norm(shapenet_mesh_vertices, axis=1))
        # partnet_meshes_max = max([np.max(np.linalg.norm(el['vertices'], axis=1)) for el in leaves])

        # factor = shapenet_mesh_max/partnet_meshes_max
        factor = shapenet_extent/partnet_extent
        for leaf in leaves:
            leaf['vertices'] = factor * leaf['vertices']
            leaf['vertices'] = leaf['vertices'] + shapenet_center

        for leaf in leaves:
            part_id = leaf['id']
            part_mesh = trimesh.Trimesh(vertices=leaf['vertices'],
                                        faces=leaf['faces'])


            # generate convex hull
            part_mesh_convex_enclosure = trimesh.convex.convex_hull(part_mesh,
                                                                    'QbB Pp Qt')
            leaf['convex_mesh'] = part_mesh_convex_enclosure

            # get bounding box
            x_min, y_min, z_min = np.min(part_mesh.vertices, axis=0)
            x_max, y_max, z_max = np.max(part_mesh.vertices, axis=0)
            # representation of the box is given by these 6 numbers!
            bbox = {'x_min': x_min,
                    'y_min': y_min,
                    'z_min': z_min,
                    'x_max': x_max,
                    'y_max': y_max,
                    'z_max': z_max}

            leaf['bbox'] = bbox

            # make and save a mesh with these parameters
            bbox_transform = np.identity(4)
            bbox_transform[:3, 3] = np.array([1/2*(x_min + x_max),
                                             1/2*(y_min + y_max),
                                             1/2*(z_min + z_max)])
            extents = np.array([x_max - x_min, y_max - y_min, z_max - z_min])
            bbox_mesh = trimesh.creation.box(extents, bbox_transform)

            leaf['bbox_mesh'] = bbox_mesh

            # export_mesh(bbox_mesh, '../bbox{}.obj'.format(part_id), file_type='obj')
            # export_mesh(part_mesh, '../foo{}.obj'.format(part_id), file_type='obj')

        # save the mesh of the part
        # os.system('cp /orion/group/ShapeNetManifold_10000/{}/{}/models/model_normalized.obj ../original_manifold.obj'.format(OBJ_CAT_ID, sid))

        # make a folder with the sid
        leaves_save_file = os.path.join(output_dir, OBJ_CAT_ID, '{}.pkl'.format(sid))
        if not os.path.exists(os.path.dirname(leaves_save_file)):
            os.makedirs(os.path.dirname(leaves_save_file))
        with open(leaves_save_file, 'wb') as f:
            pickle.dump(leaves, f)




    # iterate through some of them and save the meshes


