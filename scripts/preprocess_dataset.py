import argparse
import os
import h5py
import numpy as np
import trimesh
from tqdm import tqdm   
from mesh_to_sdf import get_surface_point_cloud
import glob


# Ensure that scripts can run on headless servers.
os.environ['PYOPENGL_PLATFORM'] = 'egl' 


def scale_to_unit_cube(verts, padding):
    '''
    Scales a given point cloud into the unit cube [-0.5, 0.5]^3 using padding.
    :param verts: The point cloud as numpy array of size [n, 3]
    :param padding: The padding 
    :return: Scaled point cloud as numpy array of size [n, 3]  
    '''
    bb_min = np.min(verts, axis = 0)
    bb_max = np.max(verts, axis = 0)
    size = np.max(bb_max - bb_min)
    bb_center = (bb_min + bb_max) / 2
    scale = (1 - padding) / size
    return (verts - bb_center) * scale

def process_mesh(file, args):
    '''
    Preprocesses a raw 3D scan. The mesh is assumed to be watertight!
    This includes the following steps: (1) Scale raw mesh into unit
    cube [-0.5, 0.5]^3, (2) extract surface (ie, discard inner parts) from 
    a raw mesh and also compute surface normals.
    :param file: The path to the raw mesh file
    :param args: The command line arguments 
    :return: A dictionary containing the sampled surface points and normals
    '''
    mesh = trimesh.load_mesh(file) 
  
    # Scale to [-0.5, 0.5]^3.
    mesh.vertices = scale_to_unit_cube(mesh.vertices, args.padding)

    # Get surface point cloud, ie, discard inner parts of mesh.
    # Also computes per-vertex normals.
    surface_pt_cloud = get_surface_point_cloud(mesh, bounding_radius = 1)
    points, normals = surface_pt_cloud.points, surface_pt_cloud.normals 

    return {
        'surf_points': points,
        'surf_normals': normals
    }

def main(args):
    if os.path.isfile(args.output):
        print(f'{args.output} already exists, override it.')
   
    dataset_file = h5py.File(args.output, 'w')

    idx = 0
    for file in tqdm(glob.glob(os.path.join(args.dataset_dir, '*.ply')) + glob.glob(os.path.join(args.dataset_dir, '*.obj'))):
        result = process_mesh(file, args)
    
        if result is not None:
            # Save result in hdf5 file.
            dataset_object = dataset_file.create_group(f'{idx}') # Use integer as key to identify an object.
            for key, value in result.items():
                dataset_object.create_dataset(key, data = value)
            idx += 1
 
if __name__ == '__main__':
    argparser = argparse.ArgumentParser()

    argparser.add_argument('dataset_dir', type = str,
                           help = 'The directory containing the raw meshes as obj or ply files.')   
    argparser.add_argument('--output', type = str,
                           help = 'The output name of the dataset file.',   
                           default = './dataset.hdf5')
    argparser.add_argument('--padding', type = float,
                           help = 'The padding to add to the unit cube.',
                           default = 0.1)
 
    args, _ = argparser.parse_known_args()

    main(args)