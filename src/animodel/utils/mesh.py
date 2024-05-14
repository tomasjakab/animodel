from scipy.spatial import cKDTree as KDTree
import trimesh
import numpy as np


def sample_surface_point(mesh, num_points):
    sample_points, indexes = trimesh.sample.sample_surface_even(mesh, count=num_points)
    while len(sample_points) < num_points:
        more_sample_points, indexes = trimesh.sample.sample_surface_even(
            mesh, count=num_points
        )
        sample_points = np.concatenate([sample_points, more_sample_points], axis=0)
    return sample_points[:num_points]


def load_mesh(file_path):
    mesh = trimesh.load_mesh(file_path)
    return mesh


def compute_trimesh_chamfer(mesh_target, mesh_reconstruction, num_mesh_samples=30000):
    """
    This function computes a symmetric chamfer distance, i.e. the sum of both chamfers.
    gt_points: trimesh.points.PointCloud of just poins, sampled from the surface (see
            compute_metrics.ply for more documentation)
    gen_mesh: trimesh.base.Trimesh of output mesh from whichever autoencoding reconstruction
            method (see compute_metrics.py for more)

    Implementation based on: https://github.com/facebookresearch/DeepSDF/blob/main/deep_sdf/metrics/chamfer.py
    """
    points_target = sample_surface_point(mesh_target, num_mesh_samples)
    points_reconstruction = sample_surface_point(mesh_reconstruction, num_mesh_samples)

    # one direction
    gen_points_kd_tree = KDTree(points_reconstruction)
    one_distances, one_vertex_ids = gen_points_kd_tree.query(points_target)
    gt_to_gen_chamfer = np.mean(one_distances)

    # other direction
    gt_points_kd_tree = KDTree(points_target)
    two_distances, two_vertex_ids = gt_points_kd_tree.query(points_reconstruction)
    gen_to_gt_chamfer = np.mean(two_distances)

    return gt_to_gen_chamfer, gen_to_gt_chamfer


def voxelize(mesh, min_cubes=100):
    voxel_length = mesh.extents.min() / min_cubes
    voxel = mesh.voxelized(voxel_length).fill()
    return voxel


def calculate_volume(mesh, min_cubes=100):
    voxel = voxelize(mesh, min_cubes=min_cubes)
    voxel_volume = voxel.volume
    aux = {"voxel": voxel}
    return voxel_volume, aux


def calculate_scale(mesh, target_volume, method="volume", min_cubes=100):
    aux = {}
    if method == "bounding_box":
        width, height, length = mesh.extents
        bounding_box_volume = width * height * length
        scale = (target_volume / bounding_box_volume) ** (1 / 3)
    elif method == "volume":
        voxel_volume, aux = calculate_volume(mesh, min_cubes=min_cubes)
        scale = (target_volume / voxel_volume) ** (1 / 3)
    return scale, aux


def remesh(mesh, min_cubes=100):
    # Voxelize the resized reconstruction mesh
    voxels = voxelize(mesh, min_cubes=min_cubes)

    # Remesh using marching cubes from the voxels
    mesh_remeshed = voxels.marching_cubes
    # The mesh has now differnt scale and translation (lost due to the voxels), align it with the original
    # Align the scales
    scale = (mesh.extents / mesh_remeshed.extents).mean()
    scale_mat = trimesh.transformations.scale_matrix(scale)
    mesh_remeshed = mesh_remeshed.apply_transform(scale_mat)
    # Align the centers
    translation = mesh.centroid - mesh_remeshed.centroid
    translation_mat = trimesh.transformations.translation_matrix(translation)
    mesh_remeshed = mesh_remeshed.apply_transform(translation_mat)
    return mesh_remeshed
