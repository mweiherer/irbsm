import torch
import random
import numpy as np


def seed_everything(seed):
    if seed < 0:
        return
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

# Source: https://stackoverflow.com/questions/29027792/get-average-value-from-list-of-dictionary.
def dict_mean(dict_list):
    mean_dict = {}
    for key in dict_list[0].keys():
        mean_dict[key] = torch.mean(torch.Tensor([d[key] for d in dict_list]), dim = 0)
    return mean_dict

def scale_to_unit_cube(verts, padding, return_transformation = False):
    '''
    Scales a given point cloud into the unit cube [-0.5, 0.5]^3 using padding.
    :param verts: The point cloud as numpy array of size n x 3
    :param padding: The padding 
    :param return_transformation: Flag indicating whether computed translation and scale should be returned
    :return: Scaled point cloud, and, optionally, computed translation and scale
    '''
    bb_min = torch.min(verts, dim = 0)[0]
    bb_max = torch.max(verts, dim = 0)[0]
    size = torch.max(bb_max - bb_min)
    bb_center = (bb_min + bb_max) / 2
    scale = (1 - padding) / size
    verts_norm = (verts - bb_center) * scale
    if return_transformation:
        return verts_norm, bb_center, scale
    return verts_norm

def rigid_landmark_alignment(src_pts, tar_pts): 
    '''
    Performs a landmark-based similarity alignment between two corresponding point sets. See:
        Shinji Umeyama, Least-squares estimation of transformation 
                        parameters between two point patterns, TPAMI'91.
    :param src_pts: The source points as torch.Tensor of size [n, 3]
    :param tar_pts: The target points as torch.Tensor of size [n, 3]
    :return: The rotation matrix R as torch.Tensor of shape [3, 3]
    :return: The translation vector t as torch.Tensor of shape [3]
    :return: The scaling factor c as scalar
    '''
    assert src_pts.shape == tar_pts.shape \
        and src_pts.dtype == tar_pts.dtype
 
    mu_src = torch.mean(src_pts, dim = 0)
    mu_tar = torch.mean(tar_pts, dim = 0)
    src_pts_white = src_pts - mu_src
    tar_pts_white = tar_pts - mu_tar
    cov_m = (src_pts_white.t() @ tar_pts_white) / src_pts.shape[0]
    U, D, V = torch.svd(cov_m)
    S = torch.eye(3).to(src_pts)
    S[2, 2] = torch.det(V @ U)
    R = V @ S @ U.t()
    c = (1 / torch.var(src_pts, dim = 0, unbiased = False).sum()) * (torch.diag(D) @ S).trace()
    t = mu_tar - c * R @ mu_src
    return R, t, c