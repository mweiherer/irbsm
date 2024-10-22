import torch
import numpy as np


def sample_evaluation_points(surf_points, surf_normals, num_eval_points):
    return_dict = {}

    # Sample points on surface to compute the chamfer and hausdorff distances.
    rand_indc = torch.randperm(surf_points.shape[0])[:num_eval_points]
    return_dict['chamfer_points'] = surf_points[rand_indc, :]
    return_dict['chamfer_normals'] = surf_normals[rand_indc, :]
    return return_dict


class ShapeReconstruction:
    def __init__(self, cfg):
        self.cfg = cfg

        self.num_surface_points = cfg['data']['num_surface_points']
        self.num_volume_points = cfg['data']['num_volume_points']
        self.num_eval_points = cfg['eval']['num_eval_points']

    def __call__(self, data):
        shape_id = data['shape_id']
        surf_points, surf_normals = data['surf_points'], data['surf_normals']
        
        # Sample on-surface points.
        rand_indc = torch.randperm(surf_points.shape[0])[:self.num_surface_points]
        sample_points, sample_normals = surf_points[rand_indc, :], surf_normals[rand_indc, :]

        # Sample points in volume, but sample more aggressively near the surface.
        vol_points_near = sample_points + torch.randn_like(sample_points) * 0.01
        vol_points_far = torch.from_numpy(2 * np.random.rand(self.num_volume_points, 3) - 1).float()

        return_dict = {
            'shape_id': shape_id,
            'surf_points': sample_points,
            'surf_normals': sample_normals,
            'vol_points_near': vol_points_near,
            'vol_points_far': vol_points_far
        }

        eval_dict = sample_evaluation_points(surf_points, surf_normals, self.num_eval_points)
        return_dict.update(eval_dict)

        return return_dict