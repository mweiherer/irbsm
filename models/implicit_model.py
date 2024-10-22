import torch
import trimesh
from tqdm import tqdm
import time
 
from utils.common import rigid_landmark_alignment, scale_to_unit_cube 
from utils.eval import make_grid_points, reconstruct_mesh


class ImplicitShapeModel:
    def __init__(self, cfg_file = './configs/deep_sdf_256.yaml', ckpt_file = './checkpoints/irbsm_256.pth', 
                 device = 'cuda', voxel_resolution = 256, chunk_size = 100_000):
        self.device = device
        self.voxel_resolution = voxel_resolution
    
        from utils.io import read_config
        from utils.setup_model import make_model
        
        ckpt = torch.load(ckpt_file)
        cfg = read_config(cfg_file)
        
        self.model = make_model(cfg, checkpoint = ckpt['model_state_dict']) 
        self.model = self.model.to(self.device)  

        grid_points = make_grid_points(self.voxel_resolution).repeat(1, 1, 1).to(self.device) 
        self.chunks = torch.split(grid_points, chunk_size, dim = 1)

        self.latent_mean = ckpt['latent_mean']
        self.latent_std = ckpt['latent_std']
        self.latent_dim = self.latent_mean.shape[0] 

        self.model_landmarks = torch.from_numpy(ckpt['landmarks']) # TODO: check if landmarks are available.
 
    def _mesh_from_latent(self, latent_code):
        '''
        Reconstructs a triangle mesh from a given latent code.
        :param latent_code: The latent code as torch.Tensor of size latent_dim
        :return: The reconstructed triangle mesh as trimesh object
        '''
        latent_code = latent_code.unsqueeze(dim = 0)

        fx_chunks = []
        for points in tqdm(self.chunks, 'Reconstruct mesh'):
            _latent_code = latent_code.unsqueeze(dim = 1).repeat(1, points.shape[1], 1).to(self.device)
            fx_i = self.model(points, _latent_code).squeeze(dim = -1).detach().cpu()
            fx_chunks.append(fx_i)
        fx = torch.cat(fx_chunks, dim = 1)
       
        fx_volume = fx.reshape(fx.shape[0], self.voxel_resolution, self.voxel_resolution, self.voxel_resolution)

        mesh = reconstruct_mesh(fx_volume[0, ...])
        return trimesh.Trimesh(vertices = mesh['vertices'], faces = mesh['faces'], vertex_normals = mesh['normals'])

    def sample(self):
        '''
        Randomly samples from the shape model.
        :return: The reconstructed triangle mesh as trimesh object
        '''
        random_latent = torch.randn(self.latent_mean.shape) * self.latent_std * 1.0 + self.latent_mean
        return self._mesh_from_latent(random_latent)

    def mean_shape(self):
        '''
        Returns the mean shape of the shape model.
        :return: The reconstructed triangle mesh as trimesh object
        '''
        return self._mesh_from_latent(self.latent_mean)  

    def reconstruct(self, point_cloud, landmarks = None, num_iterations = 5_000, num_samples_per_iteration = 1_000, latent_weight = 0.01, verbose = False):
        '''
        Reconstructs a surface mesh from a given point cloud using latent code optimization.
        :param point_cloud: The input point cloud as torch.Tensor of size [n, 3]
        :param landmarks: Optinal landmarks to rigidly align the point cloud with the model before reconstruction
        :param num_iterations: The number of optimization iterations
        :param num_samples_per_iteration: The number of points to subsample from the point cloud per iteration
        :param latent_weight: The weight of the latent regularization term
        :param verbose: Whether to print optimization progress
        :return: The reconstructed triangle mesh as trimesh object
        '''
        if landmarks is not None:
            if verbose: print('Rigidly align point cloud with model.')  
           
            R, t, c = rigid_landmark_alignment(landmarks, self.model_landmarks)
            point_cloud = (point_cloud * c) @ R.T + t
        
        if verbose: print('Scale mesh to unit cube, [-1, 1]^3.')
  
        # Scale point cloud to unit cube, [-0.5, 0.5]^3. Save transformation, so that we can later re-transform
        # the reconstructed mesh back to the original scale.
        point_cloud, pt_center, pt_scale = scale_to_unit_cube(point_cloud, padding = 0.1, return_transformation = True)
        point_cloud *= 2.0 # Scale to [-1, 1]^3

        point_cloud = point_cloud.unsqueeze(dim = 0).to(self.device)
      
        latent_code = torch.zeros([1, 1, self.latent_dim], device = self.device, requires_grad = True)   
        optimizer = torch.optim.Adam([latent_code], lr = 1e-3)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size = 1_000, gamma = 0.5)

        print(f'Reconstruct point cloud with {point_cloud.shape[1]} points.')

        start_time = time.time()
        for i in tqdm(range(num_iterations)):
            optimizer.zero_grad()

            # Subsample point cloud to num_samples_per_iteration points.
            rand_indc = torch.randperm(point_cloud.shape[1])[:num_samples_per_iteration]
            points = point_cloud[:, rand_indc, :].float()
            _latent_code = latent_code.repeat(1, points.shape[1], 1)

            pred_sdf = self.model(points, _latent_code)
            
            surface_loss = torch.abs(pred_sdf).mean()
            latent_norm = (torch.norm(latent_code, dim = -1) ** 2).mean()
            total_loss = surface_loss + latent_weight * latent_norm 

            if verbose: 
                print(f'Iteration {i}: surface_loss={surface_loss.item():.5f}, latent_norm={latent_norm.item():.5f},', \
                      f'total_loss={total_loss.item():.5f}, lr={scheduler.get_last_lr()[0]:.5f}')
            
            total_loss.backward()

            optimizer.step()
            scheduler.step()
        total_time = time.time() - start_time  

        if verbose: print(f'Latent code optimization took {total_time:.2f} seconds.')

        mesh = self._mesh_from_latent(latent_code.squeeze())

        if verbose: print('Re-transform mesh back to original scale.')
      
        # Re-transform mesh back to original scale.
        mesh.vertices = mesh.vertices / (2 * pt_scale.numpy())
        mesh.vertices += pt_center.numpy()  

        if landmarks is not None:   
            if verbose: print('Undo rigid alignment and re-transform back to original coordinate system.')  

            mesh.vertices = (((torch.from_numpy(mesh.vertices) - t) @ R) / c).numpy()

        return mesh