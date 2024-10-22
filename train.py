import torch
import argparse
import wandb

from utils.common import seed_everything
from utils.io import cond_mkdir, read_config
from utils.setup_model import make_trainer


def main(args, cfg):
    if torch.cuda.is_available():
        device = 0
    else:
        print('Training requires a gpu.'); exit()

    seed_everything(cfg['misc']['manual_seed'])

    # If a checkpoint has been specified, load it.
    wandb_dir = './wandb/'; cond_mkdir(wandb_dir)

    if args.ckpt:
        checkpoint = torch.load(args.ckpt, map_location = {'cuda:0': f'cuda:{device}'})
        run_id = args.ckpt.split('/')[-2]
        print(f'Loaded checkpoint from {args.ckpt}. Resuming run {run_id}.')  
        wandb.init(dir = wandb_dir, config = cfg, project = 'iRBSM', resume = 'must', id = run_id)
    else: 
        checkpoint = None
        wandb.init(dir = wandb_dir, config = cfg, project = 'iRBSM')

    trainer = make_trainer(cfg)
    trainer.init_training(device, checkpoint)

    print(f'Start training for {args.num_epochs} epochs.')

    trainer.train(num_epochs = args.num_epochs)
 

if __name__ == '__main__':
    argparser = argparse.ArgumentParser()
    
    argparser.add_argument('config', type = str, 
                           help = 'The path to the config file.')
    argparser.add_argument('--ckpt', type = str,
                           help = 'The path to the checkpoint file. If not given, start training from scratch.', default = '')
    argparser.add_argument('--num_epochs', type = int,
                           help = 'Number of epochs to train.', default = 10_000)

    args, _ = argparser.parse_known_args()
    cfg = read_config(args.config)

    main(args, cfg)