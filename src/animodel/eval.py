"""
Run this script to evaluate the 3D reconstruction results.

Example usage:
python eval_3d.py configs=configs/base.yaml [ARGS]

Based on evaluation from DOVE: Learning Deformable 3D Objects by Watching Videos, IJCV 2023.

Author: Tomas Jakab <tomj@robots.ox.ac.uk>
"""

import os
import os.path as osp
from concurrent.futures import ProcessPoolExecutor
from glob import glob

import numpy as np
import pandas as pd
import trimesh
from tqdm import tqdm

import animodel.utils.mesh as utils_mesh
import animodel.utils.misc as utils_misc
from animodel.utils.misc import Timer

# Logger configuration
logger = utils_misc.init_logger(logging_level="INFO")


def create_header(name, df):
    if name:
        name = name + " "
    return (
        f"mean {name}Chamfer Distance: {df['bidir'].mean():.6f} +- {df['bidir'].std():.6f}\t"
        f"to recon: {df['target_to_reconstruction'].mean():.6f} +- {df['target_to_reconstruction'].std():.6f}\t"
        f"to target: {df['reconstruction_to_target'].mean():.6f} +- {df['reconstruction_to_target'].std():.6f}\t"
    )


def print_results(title, df):
    s = f"{title}\n"
    s += "\tMean Chamfer Distances\n"
    s += f"\t\tBidirectional: {df['bidir'].mean():.6f} +- {df['bidir'].std():.6f}\n"
    s += f"\t\tTarget to reconstruction: {df['target_to_reconstruction'].mean():.6f} +- {df['target_to_reconstruction'].std():.6f}\n"
    s += f"\t\tReconstruction to target: {df['reconstruction_to_target'].mean():.6f} +- {df['reconstruction_to_target'].std():.6f}\n"
    print(s)


def write_result(
    file_path, chamfer_distances, chamfer_distances_scaled, transform_mat, cost
):
    scale, shear, angles, trans, persp = trimesh.transformations.decompose_matrix(
        transform_mat
    )
    with open(file_path, "w") as f:
        f.write(f"chamfer distance:\n")
        f.write(f'total: {chamfer_distances["bidir"]}\n')
        f.write(f'to recon: {chamfer_distances["target_to_reconstruction"]}\n')
        f.write(f'to target: {chamfer_distances["reconstruction_to_target"]}\n\n')

        f.write(f"scaled chamfer distance:\n")
        f.write(f'total: {chamfer_distances_scaled["bidir"]}\n')
        f.write(f'to recon: {chamfer_distances_scaled["target_to_reconstruction"]}\n')
        f.write(
            f'to target: {chamfer_distances_scaled["reconstruction_to_target"]}\n\n'
        )

        f.write(f"cost: {cost}\n\n")

        f.write(f"transformation matrix:\n{transform_mat}\n\n")

        f.write(f"scale: {scale}\n")
        f.write(f"shear: {shear}\n")
        f.write(f"rotation angle: {angles}\n")
        f.write(f"translation: {trans}\n")
        f.write(f"perspective transform: {persp}")


def convert_rot_mats(rot_mats):
    # convert to radians
    rot_mats = np.array(rot_mats) / 180 * np.pi
    rot_mats = [
        trimesh.transformations.euler_matrix(*rot_mat, "sxyz") for rot_mat in rot_mats
    ]
    rot_mat = trimesh.transformations.concatenate_matrices(*rot_mats)
    return rot_mat


def run_icp(
    mesh_reconstruction,
    mesh_target,
    num_points_align=1000,
    max_iterations=100,
    cost_threshold=0,
    scale=False,
):
    # icp alignment with evenly sampled points from both meshes
    # sample points from reconstruction
    reconstruction_points = utils_mesh.sample_surface_point(
        mesh_reconstruction, num_points_align
    )
    # sample points from target
    target_points = utils_mesh.sample_surface_point(mesh_target, num_points_align)
    # align reconstruction to target
    with Timer("icp", logger.debug):
        transform_mat, _, cost = trimesh.registration.icp(
            reconstruction_points,
            target_points,
            threshold=cost_threshold,
            max_iterations=max_iterations,
            reflection=False,
            translation=True,
            scale=scale,
        )
    mesh_reconstruction = mesh_reconstruction.copy()
    mesh_reconstruction.apply_transform(transform_mat)
    return mesh_reconstruction, transform_mat, cost


def calculate_chamfer_distances(mesh_target, mesh_reconstruction, num_points):
    (
        target_to_reconstruction,
        reconstruction_to_target,
    ) = utils_mesh.compute_trimesh_chamfer(mesh_target, mesh_reconstruction, num_points)
    bidir = (target_to_reconstruction + reconstruction_to_target) / 2.0
    chamfer_distances = {
        "bidir": bidir,
        "target_to_reconstruction": target_to_reconstruction,
        "reconstruction_to_target": reconstruction_to_target,
    }

    # compute size-normalized chamfer distance
    # assuming that we want the gt targetned object to fit in a unit cube
    scale = 1 / mesh_target.extents.max()
    chamfer_distances_scaled = {k: v * scale for k, v in chamfer_distances.items()}

    return chamfer_distances, chamfer_distances_scaled


def compute_for_sample(
    target_file_path,
    reconstruction_file_path,
    num_points_align=1000,
    max_iterations=100,
    cost_threshold=0,
    num_points_chamfer=30000,
    min_cubes=100,
    icp_scale=False,
    best_rotate_sample=True,
    out_folder=None,
):
    """
    Assumes Y+ is up
    """
    # 1. Load reconstruction and target meshes
    with Timer("load_mesh_target", logger.debug):
        mesh_target = utils_mesh.load_mesh(target_file_path)
    with Timer("load_mesh_reconstruction", logger.debug):
        mesh_reconstruction = utils_mesh.load_mesh(reconstruction_file_path)

    # 2. Convert them into voxels to calculate volume of each
    with Timer("calculate_volume", logger.debug):
        target_volume, _ = utils_mesh.calculate_volume(mesh_target, min_cubes=min_cubes)

    # 3. Resize the reconstruction mesh to the target mesh based on the volume
    with Timer("calculate_scale", logger.debug):
        scale, _ = utils_mesh.calculate_scale(
            mesh_reconstruction, target_volume, method="volume", min_cubes=min_cubes
        )

    scale_mat = trimesh.transformations.scale_matrix(scale)
    mesh_reconstruction = mesh_reconstruction.apply_transform(scale_mat)

    # 4.-5. Remeshing reconstruction: Voxelize the resized reconstruction mesh & Remesh using marching cubes from the voxels
    with Timer("remesh", logger.debug):
        mesh_reconstruction = utils_mesh.remesh(
            mesh_reconstruction, min_cubes=min_cubes
        )

    # 6. Run ICP to align the reconstruction and target meshes
    run_icp_fn = lambda mesh_reconstruction: run_icp(
        mesh_reconstruction,
        mesh_target,
        num_points_align=num_points_align,
        max_iterations=max_iterations,
        cost_threshold=cost_threshold,
        scale=icp_scale,
    )
    with Timer("run_icp", logger.debug):
        mesh_reconstruction, transform_mat, cost = run_icp_fn(mesh_reconstruction)

    # 7. Calculate the metrics
    # compute chamfer distances
    with Timer("compute_trimesh_chamfer", logger.debug):
        chamfer_distances, chamfer_distances_scaled = calculate_chamfer_distances(
            mesh_target, mesh_reconstruction, num_points_chamfer
        )

    # 8. Rotate the reconstruction mesh by 180 degrees and repeat 7-8.
    if best_rotate_sample:
        mesh_reconstruction_rot = mesh_reconstruction.copy()
        mesh_reconstruction_rot.apply_transform(convert_rot_mats([[0, 180, 0]]))
        mesh_reconstruction_rot, transform_mat_rot, cost_rot = run_icp_fn(
            mesh_reconstruction_rot
        )
        chamfer_distances_rot, chamfer_distances_scaled_rot = (
            calculate_chamfer_distances(
                mesh_target, mesh_reconstruction_rot, num_points_chamfer
            )
        )
        # 9. Pick the best alignment
        if chamfer_distances_rot["bidir"] < chamfer_distances["bidir"]:
            mesh_reconstruction = mesh_reconstruction_rot
            transform_mat = transform_mat_rot
            cost = cost_rot
            chamfer_distances = chamfer_distances_rot
            chamfer_distances_scaled = chamfer_distances_scaled_rot

    # 10. Save the results
    file_name = osp.basename(target_file_path)

    # export aligned
    if out_folder is not None:
        with Timer("export_aligned", logger.debug):
            _ = mesh_reconstruction.export(
                osp.join(out_folder, file_name.replace(".obj", "_aligned.obj"))
            )
        # export target
        with Timer("export_target", logger.debug):
            _ = mesh_target.export(
                osp.join(out_folder, file_name.replace(".obj", "_target.obj"))
            )

        results_file_path = osp.join(
            out_folder,
            file_name.replace(".obj", "_scores.txt"),
        )
        write_result(
            results_file_path,
            chamfer_distances,
            chamfer_distances_scaled,
            transform_mat,
            cost,
        )

    logger.debug(
        f'{file_name} - align_cost: {cost:.6f}, chamfer_dist: {chamfer_distances["bidir"]:.6f}, chamfer_dist_scaled: {chamfer_distances_scaled["bidir"]:.6f}'
    )

    return chamfer_distances, chamfer_distances_scaled


def _compute_for_sample_wrapper(reconstruction_file_path, target_file_path, params):
    return compute_for_sample(
        target_file_path,
        reconstruction_file_path,
        **params,
    )


def run(
    reconstructions_folder,
    targets_folder,
    out_folder=None,
    num_points_align=8000,
    max_iterations=200,
    cost_threshold=0,
    num_points_chamfer=60000,
    min_cubes=100,
    num_targets=None,
    target_suffix=".obj",
    icp_scale=False,
    best_rotate_sample=True,
    num_workers=0,
    **kwargs,
):
    if kwargs != {}:
        logger.warning(f"Unused arguments: {kwargs}")

    logger.info(f"loading reconstructions from {reconstructions_folder}")
    logger.info(f"loading targets from {targets_folder}")

    reconstructions_list = sorted(glob(osp.join(reconstructions_folder, "*.obj")))
    targets_list = sorted(glob(osp.join(targets_folder, "*" + target_suffix)))

    if num_targets is not None:
        num_targets = int(num_targets)
        reconstructions_list = reconstructions_list[:num_targets]
        targets_list = targets_list[:num_targets]

    assert len(reconstructions_list) == len(
        targets_list
    ), f"number of reconstructions {len(reconstructions_list) } and targets {len(targets_list)} must match"
    logger.info(f"evaluating {len(reconstructions_list)} reconstructions")

    if out_folder is not None:
        os.makedirs(out_folder, exist_ok=True)
        logger.info(f"writing results to {out_folder}")

    params = {
        "num_points_align": num_points_align,
        "max_iterations": max_iterations,
        "cost_threshold": cost_threshold,
        "num_points_chamfer": num_points_chamfer,
        "icp_scale": icp_scale,
        "out_folder": out_folder,
        "best_rotate_sample": best_rotate_sample,
        "min_cubes": min_cubes,
    }

    # 1. Compute chamfer distances for all the samples
    all_chamfer_distances = []
    all_chamfer_distances_scaled = []

    logger.info(f"using {num_workers} workers")
    if num_workers > 0:
        executor = ProcessPoolExecutor(max_workers=num_workers)
        map_fn = executor.map
    else:
        executor = None
        map_fn = map

    param_list = [params] * len(reconstructions_list)
    computation_results = map_fn(
        _compute_for_sample_wrapper, reconstructions_list, targets_list, param_list
    )
    for chamfer_distances, chamfer_distances_scaled in tqdm(
        computation_results, total=len(reconstructions_list)
    ):
        all_chamfer_distances.append(chamfer_distances)
        all_chamfer_distances_scaled.append(chamfer_distances_scaled)

    if executor is not None:
        executor.shutdown()

    # 2. Get the stats and save them
    all_chamfer_distances = pd.DataFrame(all_chamfer_distances)
    all_chamfer_distances_scaled = pd.DataFrame(all_chamfer_distances_scaled)

    header = create_header("", all_chamfer_distances) + create_header(
        "scaled", all_chamfer_distances_scaled
    )
    # concatenate the two dataframes
    all_chamfer_distances_array = np.concatenate(
        [all_chamfer_distances.to_numpy(), all_chamfer_distances_scaled.to_numpy()],
        axis=1,
    )
    if out_folder is not None:
        txt_result_path = osp.join(out_folder, "all_scores.txt")
        np.savetxt(
            txt_result_path,
            all_chamfer_distances_array,
            fmt="%.6f",
            delimiter="\t",
            header=header,
        )
        logger.info(f"saved results to {txt_result_path}")

    # Print the results
    print()
    print("Evaluation is complete.")
    print()
    print("Results")
    print("-------")
    print_results("In meters", all_chamfer_distances)
    print()
    print_results(
        "Scaled - target scaled to fit inside a unit cube", all_chamfer_distances_scaled
    )


if __name__ == "__main__":
    from animodel.utils.misc import parse_configs_and_instantiate

    parse_configs_and_instantiate("__main__.run")
