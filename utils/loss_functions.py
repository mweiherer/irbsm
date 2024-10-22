import torch

from utils.diff_operators import gradient


def igr_loss(pred_sdf, points, surf_normals, latent_code, alpha):
    surf_sdf = pred_sdf[:, :surf_normals.shape[1], :]
    vol_sdf = pred_sdf[:, surf_normals.shape[1]:, :]

    grad_sdf = gradient(pred_sdf, points)
    surf_grad = grad_sdf[:, :surf_normals.shape[1], :]

    surface_loss = torch.abs(surf_sdf).mean()
    normal_loss = torch.norm(surf_grad - surf_normals, dim = -1).mean()
    
    eikonal_loss = torch.abs(torch.norm(grad_sdf, dim = -1) - 1).mean()
    volume_loss = torch.exp(-alpha * torch.abs(vol_sdf)).mean()

    latent_norm = (torch.norm(latent_code, dim = -1) ** 2).mean()
    
    return {
        'surface_loss': surface_loss,
        'normal_loss': normal_loss,
        'eikonal_loss': eikonal_loss,
        'volume_loss': volume_loss,
        'latent_norm': latent_norm
    }