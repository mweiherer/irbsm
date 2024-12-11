import torch
import point_cloud_utils as pcu

from utils.io import read_landmarks, read_pdm
from utils.io import cond_mkdir 


def reconstruct(rbsm, point_cloud, landmarks, output_dir):
    device = 'cuda' # or 'cpu' if you want to run on a CPU.
    num_iterations = 2_000; num_samples_per_iteration = 1_000

    reconstruction = rbsm.reconstruct(point_cloud, landmarks, 
                                      device, num_iterations, 
                                      num_samples_per_iteration,
                                      verbose = False)
    
    reconstruction.export(f'{output_dir}/reconstruction.ply')  


if __name__ == '__main__':
    model_path = './rbsm-0122.h5'
    point_cloud_path = './point_cloud.ply'
    landmarks_path = './landmarks.csv'
    output_dir = './output'
    
    cond_mkdir(output_dir) 

    rbsm = read_pdm(model_path)
    point_cloud = torch.from_numpy(pcu.load_mesh_v(point_cloud_path)).double()
    landmarks = read_landmarks(landmarks_path)

    reconstruct(rbsm, point_cloud, landmarks, output_dir)