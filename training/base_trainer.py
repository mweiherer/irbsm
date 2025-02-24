from abc import ABC
import torch
import os
import wandb
from tqdm import tqdm
import numpy as np
from pycg import vis, render

from utils.common import dict_mean
from utils.io import cond_mkdir


class BaseTrainer(ABC):
    '''
    Abstract base class that takes care of training a given model. Every new trainer class should inherit
    from this class and override the following methods: train_step(), compute_loss(), and eval_step(). 
    You don't have to touch the rest!
    :param model: The model that should be trained
    :param optimizer: The optimizer used to train the model
    :param train_loader: The dataloader holding the training data
    :param val_loader: The dataloader holding the validation data
    :param test_loader: The dataloader holding the test data
    :param latent_optimizer: The optimizer used to train the latent codes
    :param latent_codes: The latent codes
    :param cfg: The config file
    '''
    def __init__(self, model, optimizer, scheduler, train_loader, val_loader, test_loader, 
                 latent_optimizer, latent_scheduler, latent_codes, cfg):
        super().__init__()
        self.model = model
        self.optimizer = optimizer
        self.scheduler = scheduler

        self.train_loader = train_loader
        self.val_loader = val_loader
        self.test_loader = test_loader

        self.latent_optimizer = latent_optimizer    
        self.latent_scheduler = latent_scheduler    
        self.latent_codes = latent_codes

        self.cfg = cfg

    def init_training(self, device, checkpoint):
        '''
        Initializes the training (optionally from a given checkpoint).
        :param device: The device it should be trained on
        :param checkpoint: Optional checkpoint from which training should start
        '''
        self.device = device
        self.model = self.model.to(self.device)

        self.checkpoint_dir = f'./wandb/iRBSM/{wandb.run.id}'
        cond_mkdir(self.checkpoint_dir)

        self.epochs_run = 0
        if checkpoint is not None:
            self._load_checkpoint(checkpoint)

        # Cache training metrics per epoch and evaluation metrics for wandb logging.
        self.train_metrics = None
        self.eval_metrics = None

    def _new_epoch(self):
        '''
        Initializes a new epoch.
        '''
        self.train_metrics = {}
        self.eval_metrics = {}
    
    def train_step(self, batch_data):
        '''
        Performs a single training step on the given batch data. This method is model-specific
        and should be adapted accordingly. 
        :param batch_data: A batch of training data
        '''
        raise NotImplementedError
    
    def compute_loss(self, batch_data):
        '''
        Computes the loss for the given batch data.
        Should return the individual loss terms as dictionary.
        :param batch_data: A batch of training data
        '''
        raise NotImplementedError
    
    def eval_step(self, batch_data, test):
        '''
        Performs a single evaluation step on the given batch data. This method is model-specific
        and should be adapted accordingly. 
        :param batch_data: A batch of evaluation data
        :param test: If true, perform inference on test data
        '''
        raise NotImplementedError
    
    def train(self, num_epochs):
        '''
        Training loop.
        :param num_epochs: The number of epochs to train
        '''
        self.model.train(); 

        for epoch in range(self.epochs_run, self.epochs_run + num_epochs):
            self._new_epoch()

            for batch_data in tqdm(self.train_loader):
                try:
                    self.train_step(batch_data)
                except Exception as exception:
                    print(exception); continue
         
            self.scheduler.step()
            self.latent_scheduler.step()    
                
            if (epoch + 1) % self.epochs_til_evaluation == 0:
                self.evaluate(); self.model.train()

            self.log_epoch(epoch)

    def evaluate(self):
        '''
        Performs model evaluation.
        '''
        self.model.eval(); print('Perform model evaluation.')

        # Randomly select one batch (= mesh) for visualization.
        rand_idx = torch.randint(len(self.val_loader), (1,))

        for idx, batch_data in enumerate(tqdm(self.val_loader)):
            mesh, metrics = self.eval_step(batch_data)
 
            if idx == rand_idx:
                # Render mesh and log it to wandb.
                mesh = vis.mesh(mesh[0][1]['vertices'], mesh[0][1]['faces'])
                rendering = render.multiview_image([mesh])
                self.to_eval_log({'metrics_dict': metrics, 'rendering': rendering})
            else:
                self.to_eval_log({'metrics_dict': metrics})

    def to_train_log(self, log_dict):
        '''
        Saves per-batch logging data to train_metrics.
        :param log_dict: Dictionary containing logging data as dictionaries or PyTorch tensors
        '''
        if self.train_metrics is None:
            raise Exception('Can not log data before new_epoch() has been called first.')

        for key, value in log_dict.items():
            if not (isinstance(value, dict) or isinstance(value, torch.Tensor)):
                raise ValueError(f'Can not log data of type {type(value)}. Only dict and torch.Tensor is allowed.')

            if key not in self.train_metrics:
                self.train_metrics[key] = []

            self.train_metrics[key].append(value)

    def to_eval_log(self, log_dict):
        '''
        Saves per-batch logging data to eval_metrics.
        :param log_dict: Dictionary containing logging data as dictionaries 
                         or an image as numpy array
        '''
        if self.eval_metrics is None:
            raise Exception('Can not log data before new_epoch() has been called first.')
        
        for key, value in log_dict.items():
            if not (isinstance(value, dict) or isinstance(value, np.ndarray)):
                raise ValueError(f'Can not log data of type {type(value)}. Only dict or image as numpy array is allowed.')
            
            if isinstance(value, dict):
                value = {f'val/{key}': value for key, value in value.items()}
            
            if key not in self.eval_metrics:
                self.eval_metrics[key] = []

            self.eval_metrics[key].append(value)

    def log_epoch(self, epoch, additional_logs = None):
        '''
        Computes per-epoch metrics from the collected per-batch train and eval data 
        and writes collected data to wandb. Also saves checkpoint to disk.
        :param epoch: The epoch
        :param additional_logs: Optional, per-epoch logs as dict that should be uploaded to wandb
        '''
        wandb_metrics = {}

        # First, we collect training metrics.
        for key, value in self.train_metrics.items():
            if all(isinstance(x, dict) for x in value):
                wandb_metrics.update(dict_mean(value))
            else: # If it's not a list of dicts, must be list of PyTorch tensors. See to_train_log().
                wandb_metrics[key] = torch.Tensor(value).mean()

        # Always log learning rate.
        wandb_metrics['train/model_lr'] = self.optimizer.param_groups[0]['lr']
        wandb_metrics['train/latent_lr'] = self.latent_optimizer.param_groups[0]['lr']
    
        if additional_logs is not None:
            wandb_metrics.update(additional_logs)

        # Next, collect evaluation metrics.
        for key, value in self.eval_metrics.items():
            if all(isinstance(x, dict) for x in value):
                wandb_metrics.update(dict_mean(value))
            elif all(isinstance(x, torch.Tensor) for x in value):
                wandb_metrics[key] = torch.Tensor(value).mean()
            else: # If it's not a list of dicts, must be an image as numpy array. See to_eval_log().
                wandb_metrics[key] = [wandb.Image(x) for x in value]

        eval_chamfer = wandb_metrics.get('val/chamfer-L1', None) 
        self._save_checkpoint(epoch, eval_chamfer)

        wandb.log(wandb_metrics, step = epoch)

        self.train_metrics = None
        self.eval_metrics = None

    def _save_checkpoint(self, epoch, eval_chamfer):
        '''
        Saves model checkpoint to disk.
        :param epoch: The epoch
        '''
        checkpoint = {
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'latent_optimizer_state_dict': self.latent_optimizer.state_dict(),  
            'latent_scheduler_state_dict': self.latent_scheduler.state_dict(),
            'latent_codes_state_dict': self.latent_codes.state_dict(),
            'epoch': epoch
        }
        torch.save(checkpoint, os.path.join(self.checkpoint_dir, 'last.pth'))
        
        if eval_chamfer is not None:
            torch.save(checkpoint, 
                       os.path.join(self.checkpoint_dir,
                                    f'epoch={str(epoch).zfill(2)}-chamfer={str(float(eval_chamfer))[:5]}.pth'))

    def _load_checkpoint(self, checkpoint):
        '''
        Loads model checkpoint from disk.
        :param checkpoint: The checkpoint
        '''  
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        self.latent_optimizer.load_state_dict(checkpoint['latent_optimizer_state_dict'])
        self.latent_scheduler.load_state_dict(checkpoint['latent_scheduler_state_dict'])
        self.latent_codes.load_state_dict(checkpoint['latent_codes_state_dict'])    
        self.epochs_run = checkpoint['epoch']
