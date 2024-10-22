import torch
import trimesh
from tqdm import tqdm
import time

from utils.common import rigid_landmark_alignment


class PointDistributionModel:
    '''
    Implementation of classical, PCA-based Point Distribution Model (PDM).
    :param reference_shape: The reference shape as trimesh object
    :param mean_shape: The mean shape as trimesh object
    :param basis: The orthonormal basis as torch.Tensor of size [num_points, num_basis]
    :param variance: The variance of the basis as torch.Tensor of size [num_basis]
    '''
    def __init__(self, reference_shape, mean_shape, basis, variance):
        self.reference_shape = reference_shape
        self._mean_shape = mean_shape
        self.mean_vec = torch.from_numpy(self._mean_shape.vertices).flatten()
    
        self.orthonormal_basis = basis
        self.variance = variance
        self.stddev = torch.sqrt(self.variance)

        self.scaled_basis = self.orthonormal_basis @ torch.diag(self.stddev)

        self.rank = self.orthonormal_basis.shape[1]
        self.number_of_points = self.reference_shape.vertices.shape[0]

        # In this order: sternal notch, nipple left and right, lower breast pole left and right.
        self.model_landmarks = [15505, 9356, 24934, 8561, 24139]

    def _mesh_from_latent(self, alpha):
        '''
        Generates a triangle mesh from a given instance vector.
        :param alpha: The instance vector as torch.Tensor of size rank
        :return: The reconstructed triangle mesh as trimesh object
        '''
        instance_vec = self.mean_vec + self.scaled_basis @ alpha
        return trimesh.Trimesh(vertices = torch.reshape(instance_vec, (int(instance_vec.shape[0] / 3), 3)), 
                               faces = self.reference_shape.faces)

    def sample(self):
        '''
        Randomly samples from the shape model.
        :return: The reconstructed triangle mesh as trimesh object
        '''
        random_alpha = torch.randn(self.rank)
        return self._mesh_from_latent(random_alpha)   
    
    def mean_shape(self):
        '''
        Returns the mean shape of the shape model.
        :return: The reconstructed triangle mesh as trimesh object
        '''
        return self._mean_shape  
    
    def reconstruct(self, point_cloud, landmarks, device = 'cuda', num_iterations = 2_000, num_samples_per_iteration = 1_000, verbose = False):
        '''
        Reconstructs a surface mesh from a given point cloud using model fitting (registration).
        :param point_cloud: The input point cloud as torch.Tensor of size [n, 3]
        :param landmarks: The landmarks as torch.Tensor of size [m, 3]
        :param device: The device to run the optimization on
        :param num_iterations: The number of optimization iterations
        :param num_samples_per_iteration: The number of points to subsample from the model per iteration
        :param verbose: Whether to print optimization progress
        :return: The reconstructed triangle mesh as trimesh object
        '''
        # Initial rigid alignment of model to point_cloud based on landmarks.
        if verbose: print('Rigidly align model to point cloud based on supplied landmarks.')

        model_landmarks = torch.from_numpy(self._mean_shape.vertices[self.model_landmarks, :])
        R, t, c = rigid_landmark_alignment(model_landmarks, landmarks)
        aligned_mean_vec = ((torch.from_numpy(self._mean_shape.vertices) * c) @ R.T + t).flatten().to(device)

        point_cloud = point_cloud.to(device)
        landmarks = landmarks.to(device)    

        alpha = torch.zeros([self.rank], device = device, requires_grad = True)
        optimizer = torch.optim.Adam([alpha], lr = 1e-2)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size = 500, gamma = 0.5)

        print(f'Fit model on point cloud with {point_cloud.shape[0]} points.')

        start_time = time.time()
        for i in tqdm(range(num_iterations)):
            optimizer.zero_grad()   

            # Get current model instance corresponding to alpha.
            current_instance = aligned_mean_vec + self.scaled_basis.to(device) @ alpha
            current_instance = torch.reshape(current_instance, (int(current_instance.shape[0] / 3), 3)).to(point_cloud)

            # Subsample model instance to num_samples_per_iteration points.
            rand_indc = torch.randperm(current_instance.shape[0])[:num_samples_per_iteration]
            points = current_instance[rand_indc, :]
            
            _, nearest_indices = torch.min(torch.cdist(points, point_cloud), dim = 1)
            nearest_neighbors = point_cloud[nearest_indices, :]
            
            surface_loss = torch.mean((points - nearest_neighbors) ** 2)
            landmark_loss = torch.mean((current_instance[self.model_landmarks, :] - landmarks) ** 2)
            latent_norm = (torch.norm(alpha, dim = -1) ** 2).mean()
            total_loss = surface_loss + 0.1 * landmark_loss + 0.01 * latent_norm

            if verbose: 
                print(f'Iteration {i}: surface_loss={surface_loss.item():.5f}, landmark_loss={landmark_loss.item():.5f},', \
                      f'latent_norm={latent_norm.item():.5f}, total_loss={total_loss.item():.5f}, lr={scheduler.get_last_lr()[0]:.5f}') 
 
            total_loss.backward()

            optimizer.step()
            scheduler.step()
        total_time = time.time() - start_time  

        if verbose: print(f'Model fitting took {total_time:.2f} seconds.')
        
        return trimesh.Trimesh(vertices = current_instance.cpu().detach().numpy(), faces = self.reference_shape.faces)