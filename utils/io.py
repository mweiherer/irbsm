import torch
import numpy as np
import h5py
import trimesh
import yaml
from pathlib import Path
import xml.etree.ElementTree as ET

from models import PointDistributionModel


def cond_mkdir(path):
    Path(path).mkdir(parents = True, exist_ok = True)

def read_config(config):
    '''
    Loads a given config file in yaml format.
    :param config: The config file in yaml format
    :return: The config file as dictionary
    '''
    with open(config, 'r') as stream:
        cfg = yaml.safe_load(stream)
    return cfg

def dtype_from_string(dtype_str, as_type = 'torch'):
    '''
    Converts dtype given as string to dtype in PyTorch or Numpy format.
    :param dtype_str: The dtype as string
    :param as_type: The target dtype, either 'torch' or 'numpy' (default 'torch')
    :return: Converted dtype
    '''
    if as_type not in ['torch', 'numpy']:
        raise ValueError(f"Unkown type '{as_type}' given. Must be either 'torch' or 'numpy'.")

    if dtype_str == 'float16':
        return torch.float16 if as_type == 'torch' else np.float16
    if dtype_str == 'float32':
        return torch.float32 if as_type == 'torch' else np.float32
    if dtype_str == 'float64':
        return torch.float64 if as_type == 'torch' else np.float64

def read_pdm(fname):
    '''
    Loads a PCA-based Point Distribution Model from a .h5 file as generated 
    by Scalismo, for example (see: https://scalismo.org).
    :param fname: Path to the .h5 file  
    :return: PointDistributionModel
    '''
    file = h5py.File(fname)
    model = file['model']
    triangulation = file['representer']['cells'][:].transpose()
   
    mean_vec = model['mean'][:]
    mean_matrix = np.reshape(mean_vec, (int(mean_vec.shape[0] / 3), 3))
    mean_shape = trimesh.Trimesh(vertices = mean_matrix, faces = triangulation)
 
    reference_matrix = np.array(file['representer']['points']).transpose()
    reference_shape = trimesh.Trimesh(vertices = reference_matrix, faces = triangulation)
    
    orthonormal_basis = torch.from_numpy(model['pcaBasis'][:])
    variance = torch.from_numpy(model['pcaVariance'][:])

    return PointDistributionModel(reference_shape, mean_shape, orthonormal_basis, variance)

def read_landmarks(landmarks_file):  
    '''
    Loads m landmark positions stored in either .pp or .csv file. 
    :param landmarks_file: Path to the file containing landmarks
    :return: Landmark positions as torch.Tensor of size [m, 3]
    ''' 
    file_extension = landmarks_file.split('/')[-1].split('.')[-1]     
    
    if file_extension == 'pp':
        tree = ET.parse(landmarks_file)
        root = tree.getroot()

        points = []
        for point in root.findall('.//point'):
            x = float(point.get('x'))
            y = float(point.get('y'))
            z = float(point.get('z'))
            points.append([x, y, z])
        return torch.from_numpy(np.array(points))
    
    if file_extension == 'csv':
        return torch.from_numpy(np.loadtxt(landmarks_file, delimiter = ','))
   
    raise ValueError('Invalid file extension. Supported extensions are .pp and .csv.')