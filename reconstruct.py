import torch
import argparse
import point_cloud_utils as pcu

from models.implicit_model import ImplicitShapeModel
from utils.io import cond_mkdir, read_landmarks


def main(args):
    if args.device not in ['cpu', 'cuda']:
        print('Invalid device.'); exit()

    if not args.ckpt:
        print('No checkpoint found.'); exit()

    irbsm = ImplicitShapeModel(cfg_file = args.config, ckpt_file = args.ckpt, device = args.device, 
                               voxel_resolution = args.voxel_resolution, chunk_size = args.chunk_size)           

    cond_mkdir(args.output_dir) 

    point_cloud = torch.from_numpy(pcu.load_mesh_v(args.point_cloud)).double()

    if args.landmarks:
        landmarks = read_landmarks(args.landmarks)
        print(f'Loaded {len(landmarks)} landmarks from {args.landmarks}.')
    else:
        landmarks = None
        print('No landmarks given. The point cloud will be reconstructed as is.')
    
    reconstruction = irbsm.reconstruct(point_cloud, landmarks, latent_weight = args.latent_weight) 

    reconstruction.export(f'{args.output_dir}/reconstruction.ply')  


if __name__ == '__main__':
    argparser = argparse.ArgumentParser()
    
    argparser.add_argument('point_cloud', type = str, 
                           help = 'Path to point cloud that should be reconstruct.')
    argparser.add_argument('--landmarks', type = str, 
                           help = 'Path to the landmarks file.')
    argparser.add_argument('--output_dir', type = str, 
                           help = 'The directory to save the samples in.',
                           default = './')
    argparser.add_argument('--latent_weight', type = float,
                           help = 'The regularization weight that penalizes deviation from the mean shape.',
                           default = 0.01)
    argparser.add_argument('--device', type = str, 
                           help = 'The device to run the model on.',
                           default = 'cuda')
    argparser.add_argument('--voxel_resolution', type = int,
                           help = 'The resolution of the voxel grid.',
                           default = 256)
    argparser.add_argument('--chunk_size', type = int,
                           help = 'Size of the chunks to split the voxel grid into.',
                           default = 100_000)
    argparser.add_argument('--config', type = str,
                           help = 'The path to the config file.', 
                           default = './configs/deep_sdf_256.yaml')
    argparser.add_argument('--ckpt', type = str,
                           help = 'The path to the checkpoint file.', 
                           default = './checkpoints/irbsm_256.pth')
    
    args, _ = argparser.parse_known_args()

    main(args)