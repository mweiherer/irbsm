import torch.nn as nn
import torch
import numpy as np


# Code adapted from https://github.com/SimonGiebenhain/NPHM/blob/main/src/NPHM/models/deepSDF.py.
class DeepSDF(nn.Module):
    '''
    DeepSDF model with skip connection and geometric initialization.
    :param cfg: The config file 
    '''
    def __init__(self, cfg):
        super(DeepSDF, self).__init__()

        input_dim = cfg['model']['input_dim']
        output_dim = cfg['model']['output_dim']
        latent_dim = cfg['model']['latent_dim']
        hidden_dim = cfg['model']['hidden_dim']
        num_layers = cfg['model']['num_layers']

        d_in = input_dim + latent_dim

        dims = [hidden_dim] * num_layers
        dims = [d_in] + dims + [output_dim]

        self.num_layers = len(dims)
        self.skip_in = [num_layers // 2]
   
        for layer in range(0, self.num_layers - 1):
            if layer + 1 in self.skip_in:
                out_dim = dims[layer + 1] - d_in
            else:
                out_dim = dims[layer + 1]

            lin = nn.Linear(dims[layer], out_dim)

            if layer == self.num_layers - 2:
                torch.nn.init.normal_(lin.weight, mean = np.sqrt(np.pi) / np.sqrt(dims[layer]), std = 0.00001)
                torch.nn.init.constant_(lin.bias, -1)

            setattr(self, "lin" + str(layer), lin)

        self.actvn = nn.Softplus(beta = 100)
       
    def forward(self, input_points, latent_code):
        '''
        Forward method of DeepSDF model.
        :param input_points: The input points as torch.Tensor of size [b, n, 3]
        :param latent_code: The latent code as torch.Tensor of size [b, n, latent_dim]
        :return: The predicted SDF values as torch.Tensor of size [b, n, 1]
        '''
        input = torch.cat([input_points, latent_code], dim = -1)
        x = input

        for layer in range(0, self.num_layers - 1):
            lin = getattr(self, "lin" + str(layer))

            if layer in self.skip_in:
                x = torch.cat([x, input], -1) / np.sqrt(2)

            x = lin(x)

            if layer < self.num_layers - 2:
                x = self.actvn(x)

        return x