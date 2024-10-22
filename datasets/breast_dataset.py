import torch
import os
import h5py
import bisect

from utils.io import dtype_from_string


class BreastDataset(torch.utils.data.Dataset):
    def __init__(self, cfg, mode, transform):
        super(BreastDataset, self).__init__()

        dataset_root = cfg['data']['dataset_root']
        studies = ['dataset']

        self.h5_paths, self.num_shapes = [], [0]

        for study in studies:
            path = os.path.join(dataset_root, study) + '.hdf5' 
            file = h5py.File(path, 'r')
            self.h5_paths.append(path)
            self.num_shapes.append(len(file.keys()) + self.num_shapes[-1])
  
        self.h5_files = []

        self.dtype = dtype_from_string(cfg['misc']['dtype'], as_type = 'numpy')

        self.transform = transform
        self.translate, self.scale = (0.0, 2.0) # To scale from [-0.5, 0.5]^3 to [-1, 1]^3.
    
    def __len__(self):
        return self.num_shapes[-1]
    
    def __getitem__(self, idx):
        if not self.h5_files:
            self.h5_files.extend([h5py.File(path, 'r') for path in self.h5_paths])

        # See https://discuss.pytorch.org/t/multiple-datasets/36928/3.
        file_idx = bisect.bisect_right(self.num_shapes, idx) - 1
        shape_idx = str(idx - self.num_shapes[file_idx])

        shape = self.h5_files[file_idx][shape_idx]  

        ret_dict = {'shape_id': idx, 'scale': [self.scale]}

        surf_pts = shape['surf_points'][:].astype(self.dtype)
        ret_dict['surf_points'] = self.scale * (torch.from_numpy(surf_pts) + self.translate)

        surf_nms = shape['surf_normals'][:].astype(self.dtype)
        ret_dict['surf_normals'] = torch.from_numpy(surf_nms)

        if self.transform is not None:
            return self.transform(ret_dict)
    
        return ret_dict