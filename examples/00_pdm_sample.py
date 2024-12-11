from utils.io import read_pdm
from utils.io import cond_mkdir 


def sample(rbsm, output_dir, num_samples):
    for i in range(num_samples):   
        random_sample = rbsm.sample()
        random_sample.export(f'{output_dir}/sample_{i}.ply')


if __name__ == '__main__':
    model_path = './rbsm-0122.h5'
    output_dir = './output'
    
    cond_mkdir(output_dir) 

    rbsm = read_pdm(model_path)
    sample(rbsm, output_dir, num_samples = 3)