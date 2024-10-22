import torch
import numpy as np
import point_cloud_utils as pcu
from skimage.measure import marching_cubes

from utils.common import dict_mean
from utils.metrics import MeshEvaluator


def make_grid_points(N):
    g1, g2, g3 = [g * 1j for g in [N, N, N]]
    marching_cubes_grid_points = np.stack([np.ravel(a) for a in
                                           np.mgrid[-1.0:1.0:g1, -1.0:1.0:g2, -1.0:1.0:g3]],
                                           axis = -1).astype(np.float32)
    return torch.from_numpy(marching_cubes_grid_points)

def reconstruct_mesh(sdf_volume):
    '''
    Reconstructs a triangle mesh from a given SDF volume using Marching Cubes. 
    :param sdf_volume: The SDF volume as torch.Tensor of size [d, d, d], where d is voxel resolution
    :return: The reconstructed triangle mesh as dictionary 
    '''
    voxel_resolution = sdf_volume.shape[0]
    voxel_size = 2.0 / (voxel_resolution - 1)
   
    try:
        vertices, faces, normals, _ = marching_cubes(sdf_volume.numpy(), level = 0.0, spacing = [voxel_size] * 3) 
    except (ValueError, RuntimeError) as e: raise e
   
    vertices -= 1.0 # Translate back to [-1, 1]^3 since MC shiftes points to [0, 2]^3.
    return {'vertices': vertices, 'faces': faces, 'normals': normals}

def compute_evaluation_metrics(fx_volume, chamfer_points, chamfer_normals, shape_ids):
    '''
    Computes evaluation metrics between predicted and ground-truth surface meshes.
    Specifically, this method computes Chamfer distance, Hausdorff distance, and normal similarity.
    :param fx_volume: The predicted SDF in the common unit grid as torch.Tensor of size [b, d, d, d], where d is voxel resolution 
    :param chamfer_points: Points on ground-truth surface from which Chamfer distance should be computed as torch.Tensor of size [b, n, 3]
    :param shape_ids: List of original shape IDs
    :return meshes: List containing tuples of shape ID and extracted surfaces as triangle meshes in trimesh format
    :return metrics: Dictionary containing computed metrics
    '''
    assert fx_volume.shape[0] == len(shape_ids)

    meshes, metrics = [], []
    for i, shape_id in enumerate(shape_ids):
        try:
            mesh = reconstruct_mesh(fx_volume[i, ...])
            meshes.append((shape_id, mesh))
        except (ValueError, RuntimeError) as e:
            print(f'Error in Marching Cubes for mesh {i}: {e} Skip it.'); continue

        # Subsample reconstructed surface mesh.
        try: # I don't know why, but sometimes interpolate_barycentric_coords(...) throws an IndexError.
            f_i_rnd, bc_i_rnd = pcu.sample_mesh_random(mesh['vertices'], mesh['faces'], chamfer_points[i, ...].shape[0]) 
            mesh_points = pcu.interpolate_barycentric_coords(mesh['faces'], f_i_rnd, bc_i_rnd, mesh['vertices'])
            mesh_normals = pcu.interpolate_barycentric_coords(mesh['faces'], f_i_rnd, bc_i_rnd, mesh['normals'])
        except IndexError:
            print(f'IndexError during subsampling of mesh {i}. Skip it.'); continue

        surf_metrics = MeshEvaluator().evaluate(mesh_points.astype(np.float32), chamfer_points[i, ...].numpy().astype(np.float32), 
                                                mesh_normals, chamfer_normals[i, ...].numpy())
    
        metrics.append(surf_metrics)
    
    return meshes, dict_mean(metrics) if len(metrics) > 0 else {}