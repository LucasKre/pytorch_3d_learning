import torch
import trimesh
from torch.utils.data import Dataset
from pytorch3Dlearning.dataset.preprocessing import MeshToPointCloud
import numpy as np
import os


class FaustToyDataset(Dataset):
    """
    Toy dataset for testing.
    """

    def __init__(self, transform=MeshToPointCloud()):
        self.mesh_dir = "data/faust/raw/"
        self.labels = torch.from_numpy(np.load("data/faust/segmentations.npz")["segmentation_labels"])
        self.transform = transform
        self.data = self.__load_meshes()
        self.data = [self.transform(mesh) for mesh in self.data]

    def __load_meshes(self):
        files = os.listdir(self.mesh_dir)
        meshes = []
        for file in files:
            mesh = trimesh.load(self.mesh_dir + file)
            meshes.append(mesh)
        return meshes

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx], self.labels
