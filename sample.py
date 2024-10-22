import argparse

from models import ImplicitShapeModel
from utils.io import cond_mkdir 


def main(args):
    if args.num_samples < 1:
        print('Number of samples must be greater than 0.'); exit()  

    if args.device not in ['cpu', 'cuda']:
        print('Invalid device.'); exit()

    if not args.ckpt:
        print('No checkpoint found.'); exit()

    irbsm = ImplicitShapeModel(cfg_file = args.config, ckpt_file = args.ckpt, device = args.device, 
                               voxel_resolution = args.voxel_resolution, chunk_size = args.chunk_size)      

    cond_mkdir(args.output_dir) 

    for i in range(args.num_samples):   
        random_sample = irbsm.sample()
        random_sample.export(f'{args.output_dir}/sample_{i}.ply')


if __name__ == '__main__':
    argparser = argparse.ArgumentParser()
    
    argparser.add_argument('num_samples', type = int, 
                           help = 'The number of samples to generate.')
    argparser.add_argument('--output_dir', type = str, 
                           help = 'The directory to save the samples in.',
                           default = './')
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