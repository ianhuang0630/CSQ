import os

import numpy as np
import torch
from torch.utils.data import Dataset
from torchvision import transforms
from torchvision.transforms import Normalize as TorchNormalize


class BaseDatasetPair(Dataset):

    def __init__(self, dataset_object, list_location, voxelizer_factory,
                 n_points_from_mesh=1000, transform=None):


        self._dataset_object = dataset_object
        print ("%d models in total ..." % (len(self._dataset_object)))

        # Number of samples to use for supervision
        self._n_points_from_mesh = n_points_from_mesh
        # Get the voxelizer to be used
        self._voxelizer = voxelizer_factory.voxelizer
        # Operations on the datapoints
        self.transform = transform

        self._input_dim = self._voxelizer.output_shape
        # 3 for the points and 3 for the normals of each face
        self._output_dim = (n_points_from_mesh, 6)

        # compiling the pairs
        self.tag2dataset_idx = {}
        for idx, element in enumerate(self._dataset_object):
            self.tag2dataset_idx[element.tag] = idx

        assert os.path.exists(list_location)
        with np.load(list_location) as triplets_f:
            triplets = triplets_f['arr_0']

        self.pairs_tags = []
        for triplet in triplets:

            if triplet[0] in self.tag2dataset_idx\
               and triplet[1] in self.tag2dataset_idx:
                self.pairs_tags.append((triplet[0], triplet[1]))
            if triplet[1] in self.tag2dataset_idx\
               and triplet[2] in self.tag2dataset_idx:
                self.pairs_tags.append((triplet[1], triplet[2]))
            if triplet[0] in self.tag2dataset_idx\
               and triplet[2] in self.tag2dataset_idx:
                self.pairs_tags.append((triplet[2], triplet[0]))


        # triplets -> all pairs


    def __len__(self):
        # size of the pairs
        return len(self.pairs_tags)

    def __getitem__(self, idx):

        tag_from, tag_to = self.pairs_tags[idx]
        idx_from = self.tag2dataset_idx[tag_from]
        idx_to = self.tag2dataset_idx[tag_to]

        m_from = self._dataset_object[idx_from]
        # print m.path_to_mesh_file
        # print m.path_to_tsdf_file
        X_from = self._voxelizer.get_X(m_from)
        mesh = m_from.groundtruth_mesh
        y_target_from = mesh.sample_faces(self._n_points_from_mesh)
        datapoint_from = (
            X_from,
            y_target_from.astype(np.float32)
        )

        m_to = self._dataset_object[idx_to]
        # print m.path_to_mesh_file
        # print m.path_to_tsdf_file
        X_to = self._voxelizer.get_X(m_to)
        mesh = m_to.groundtruth_mesh
        y_target_to = mesh.sample_faces(self._n_points_from_mesh)
        datapoint_to = (
            X_to,
            y_target_to.astype(np.float32)
        )

        # Store the dimentionality of the input tensor and the y_target tensor
        self._input_dim = datapoint_from[0].shape
        self._output_dim = datapoint_from[1].shape

        if self.transform:
            datapoint_from = self.transform(datapoint_from)
        if self.transform:
            datapoint_to = self.transform(datapoint_to)

        return {"X_from": datapoint_from[0],
                "y_from": datapoint_from[1],
                "X_to": datapoint_to[0],
                "y_to": datapoint_to[1]}

    def get_random_datapoint(self, idx=None):
        if idx is None:
            idx = np.random.choice(np.arange(self.__len__()))
        return self.__getitem__(idx)

    @property
    def input_dim(self):
        return self._input_dim

    @property
    def output_dim(self):
        return self._output_dim



class BaseDataset(Dataset):
    """Dataset is a wrapper for all datasets we have
    """
    def __init__(self, dataset_object, voxelizer_factory,
                 n_points_from_mesh=1000, transform=None):
        """
        Arguments:
        ---------
            dataset_object: a dataset object that can be either ShapeNetObject
                            or SurrealBodiesObject
            voxelizer_factory: a factory that creates a voxelizer object to
                               voxelizes the ground-truth mesh of a
                               ShapeNetModel instance
            n_points_from_mesh: int, the number of points to be sampled from
                                the groundtruth mesh
            transform: Callable that applies a transform to a sample
        """
        self._dataset_object = dataset_object

        # NOTE: assuming that dataset has tag attribute
        self._tags = dataset_object._tags

        print ("%d models in total ..." % (len(self._dataset_object)))

        # Number of samples to use for supervision
        self._n_points_from_mesh = n_points_from_mesh
        # Get the voxelizer to be used
        self._voxelizer = voxelizer_factory.voxelizer
        # Operations on the datapoints
        self.transform = transform

        self._input_dim = self._voxelizer.output_shape
        # 3 for the points and 3 for the normals of each face
        self._output_dim = (n_points_from_mesh, 6)

    def __len__(self):
        return len(self._dataset_object)

    def __getitem__(self, idx):

        model_tag = self._tags[idx]

        m = self._dataset_object[idx]
        # print m.path_to_mesh_file
        # print m.path_to_tsdf_file
        X = self._voxelizer.get_X(m)
        mesh = m.groundtruth_mesh
        y_target = mesh.sample_faces(self._n_points_from_mesh)

        datapoint = (
            X,
            y_target.astype(np.float32)
        )

        # Store the dimentionality of the input tensor and the y_target tensor
        self._input_dim = datapoint[0].shape
        self._output_dim = datapoint[1].shape

        if self.transform:
            datapoint = self.transform(datapoint)

        return model_tag, datapoint

    def get_random_datapoint(self, idx=None):
        if idx is None:
            idx = np.random.choice(np.arange(self.__len__()))
        return self.__getitem__(idx)

    @property
    def input_dim(self):
        return self._input_dim

    @property
    def output_dim(self):
        return self._output_dim


class MeshParserDataset(BaseDataset):
    """MeshDataset is a class that simply parses meshes
    """
    def __init__(self, dataset_object):
        self._dataset_object = dataset_object
        print ("%d models in total ..." % (len(self._dataset_object)))

    def __getitem__(self, idx):
        m = self._dataset_object[idx]
        mesh = m.groundtruth_mesh
        return (mesh, m.tag)

    @property
    def input_dim(self):
        raise NotImplementedError()

    @property
    def output_dim(self):
        raise NotImplementedError()


class DatasetWithTags(BaseDataset):
    def __getitem__(self, idx):
        m = self._dataset_object[idx]
        X = self._voxelizer.get_X(m)
        mesh = m.groundtruth_mesh
        y_target = mesh.sample_faces(self._n_points_from_mesh).T

        datapoint = (
            X,
            y_target.astype(np.float32).T
        )

        # Store the dimentionality of the input tensor and the y_target tensor
        self._input_dim = datapoint[0].shape
        self._output_dim = datapoint[1].shape

        if self.transform:
            datapoint = self.transform(datapoint)

        return datapoint + (m.tag.split("/")[-1],)


class DatasetWithTagsAndFaces(BaseDataset):
    def __getitem__(self, idx):
        m = self._dataset_object[idx]
        X = self._voxelizer.get_X(m)
        mesh = m.groundtruth_mesh
        y_target = mesh.sample_faces(self._n_points_from_mesh).T

        datapoint = (
            X,
            y_target.astype(np.float32).T
        )

        # Store the dimentionality of the input tensor and the y_target tensor
        self._input_dim = datapoint[0].shape
        self._output_dim = datapoint[1].shape

        if self.transform:
            datapoint = self.transform(datapoint)

        tag = m.tag.split("/")[-1]
        return datapoint + (tag, m.path_to_mesh_file, mesh.points)


class ToTensor(object):
    """Convert ndarrays in sample to Tensors."""
    def __call__(self, sample):
        X, y_target = sample

        # Do some sanity checks to ensure that the inputs have the appropriate
        # dimensionality
        # assert len(X.shape) == 4
        # return (torch.from_numpy(X), torch.from_numpy(y_target).float())
        return (torch.Tensor(X), torch.Tensor(y_target).float())

class Normalize(object):
    """Normalize image based based on ImageNet."""
    def __call__(self, sample):
        X, y_target = sample
        X = X.float()

        # The normalization will only affect X
        normalize = TorchNormalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
        X = X.float() / 255.0
        return (normalize(X), y_target)


def compose_transformations(voxelizer_factory):
    transformations = [ToTensor()]
    if voxelizer_factory == "image":
        transformations.append(Normalize())

    return transforms.Compose(transformations)


def get_dataset_type(loss_type):
    return {
        "euclidean_dual_loss": BaseDataset,
        "euclidean_dual_loss_pair": BaseDatasetPair
    }[loss_type]
