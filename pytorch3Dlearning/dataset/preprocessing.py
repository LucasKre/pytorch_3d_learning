import torch
import trimesh


class MeshToPointCloud(object):
    """
    Convert a mesh to a point cloud.
    """

    def __init__(self, use_fae_center=True, return_normals=False):
        self.use_fae_center = use_fae_center
        self.return_normals = return_normals

    def __call__(self, mesh):
        if self.use_fae_center:
            pos = mesh.vertices - mesh.vertices.mean(axis=0)
            return pos, mesh.face_normals
        else:
            pos = mesh.vertices
            return pos, mesh.vertex_normals


class MoveToOriginTransform(object):
    """
    Move the point cloud to the origin.
    """

    def __init__(self):
        pass

    def __call__(self, pos, x):
        pos = pos - pos.mean(dim=0)
        return pos, x


class NormalizeMesh(object):
    """Normalize the point cloud to fit within a unit sphere centered at the origin."""

    def __init__(self):
        pass

    def __call__(self, pos, x):
        center = torch.mean(pos, dim=0)
        distances = torch.linalg.norm(pos - center, dim=1)
        max_distance = torch.max(distances)

        # Calculate the scaling factor to fit within a unit sphere
        scale_factor = 1.0 / max_distance

        scaled_pos = pos * scale_factor
        return scaled_pos, x
