# iRBSM: A Deep Implicit 3D Breast Shape Model

**[Paper (arXiv)](https://arxiv.org/abs/2412.13244) | [Project page](https://rbsm.re-mic.de/implicit/)** 

[Maximilian Weiherer](https://mweiherer.github.io)$^{1,2}$, Antonia von Riedheim $^3$, [Vanessa Brébant](https://www.linkedin.com/in/vanessa-brebant-0a391843/)$^3$, [Bernhard Egger](https://eggerbernhard.ch)$^1$, [Christoph Palm](https://re-mic.de/en/head/)$^2$\
$^1$ Friedrich-Alexander-Universität Erlangen-Nürnberg\
$^2$ OTH Regensburg\
$^3$ University Hospital Regensburg

Official implementation of the paper "iRBSM: A Deep Implicit 3D Breast Shape Model".

This repository contains code for the implicit Regensburg Breast Shape Model (iRBSM).
Along with the inference code (sampling from our model and reconstructing point clouds), we also provide the code that has been used to train our model. 
Unfortunately, however, we can't make our training data public.

Abstract:
*We present the first deep implicit 3D shape model of the female breast, building upon and improving the recently proposed Regensburg Breast Shape Model (RBSM).
Compared to its PCA-based predecessor, our model employs implicit neural representations; hence, it can be trained on raw 3D breast scans and eliminates the need for computationally demanding non-rigid registration -- a task that is particularly difficult for feature-less breast shapes. 
The resulting model, dubbed iRBSM, captures detailed surface geometry including fine structures such as nipples and belly buttons, is highly expressive, and outperforms the RBSM on different surface reconstruction tasks.
Finally, leveraging the iRBSM, we present a prototype application to 3D reconstruct breast shapes from just a single image.*

## Setup
We're using Python 3.9, PyTorch 2.0.1, and CUDA 11.7.
To install all dependencies within a conda environment, simply run: 
```
conda env create -f environment.yaml
conda activate irbsm
```
This may take a while.

You can download the iRBSM [here](https://rbsm.re-mic.de/implicit/).
After downloading, make sure to place the `.pth` file in the `./checkpoints` folder. 

## Usage

### Randomly Sampling From the Model
To produce random samples from the iRBSM, use
```
python sample.py <number-of-samples>
```
This generates `<number-of-samples>` random breast shapes which are saved as `.ply` files.

Optional arguments:
- `--output_dir`: The output directory where to save samples. Default: `./`.
- `--device`: The device the model should run on. Default: `cuda`.
- `--voxel_resolution`: The resolution of the voxel grid on which the implicit model is evaluated. Default: `256`.
- `--chunk_size`: The size of the chunks the voxel grid should be split into. Default: `100_000`. If you have a small GPU with only little VRAM, lower this number.

### Fitting the Model to a Point Cloud
To reconstruct a point cloud using the iRBSM, type
```
python reconstruct.py <path-to-point-cloud>
```
Optional arguments:
- `--landmarks`: Path to a file containing landmarks. Please see details below.
- `--output_dir`: The output directory where to save the reconstruction. Default: `./`.
- `--latent_weight`: The regularization weight that penalizes the L2 norm of the latent code. Default: `0.01`. If you have noisy inputs, increase this number. 
- `--device`: The device the model should run on. Default: `cuda`.
- `--voxel_resolution`: The resolution of the voxel grid on which the implicit model is evaluated. Default: `256`.
- `--chunk_size`: The size of the chunks the voxel grid should be split into. Default: `100_000`. If you have a small GPU with only little VRAM, lower this number.

#### Providing Landmarks 
Whenever the orientation of the given point cloud differes significantly from the model's orientation, you should first *roughly* align both coordinate systems (this is necessary because our model is not invariant to rigid transformations).
The easiest way to achieve this is by providing certain landmark positions. 
These points can then be used to rigidly align the given point cloud to the model.
Please provide the following landmarks in *exactly* this order:
1. Sternal notch
2. Belly button
3. Left nipple (from the patient's perspective; so it's actually *right* from your perspective!)
4. Right nipple (from the patient's perspective; so it's actually *left* from your perspective!)

We recommend using [MeshLab](https://www.meshlab.net)'s PickPoints (PP) tool, which allows you to export selected point positions as XML file with `.pp` extension. 
You can directly pass this file into `reconstruct.py`.
Alternatively, you can use your favorite point picker tool and pass points as comma-separated `.csv` file.
Lastly, we also provide a simple application to interactively select points, just run
```
python scripts/pick_landmarks.py <path-to-point-cloud>
```
Please also see the README file in `./scripts`.

## Training Our Model on Your Own Data

### Preprocess Your Data
To train our model on your own watertight 3D breast scans, you first need to bring your data into the file format we're using (we're expecting training data to be stored in `.hdf5` files). 
The following script does that for you; it first scales raw meshes into the unit cube, and then optionally discards inner structures. 
Finally, it procudes a ready-to-use `.hdf5` file, that you can later plugin into our training pipeline.
Simply type
```
python scripts/preprocess_dataset.py <path-to-your-scans>
```
Optional arguments:
- `--output`: The name of the output `.hdf5` file. Default: `./dataset.hdf5`.
- `--padding`: The padding to add to the unit cube. Default: `0.1`.

After preprocessing, make sure to place the resulting `.hdf5` file(s) in the `./data` folder.

### Train the Model
To train the model, type
```
python train.py configs/deep_sdf_256.yaml
```
For training, you'd also need a wandb account. 
To log in to your account, simply type `wandb login` and follow the instructions.

## Citation 
If you use the iRBSM, please cite
```
@misc{weiherer2024irbsm,
    title={iRBSM: A Deep Implicit 3D Breast Shape Model},
    author={Weiherer, Maximilian and von Riedheim, Antonia and Brébant, Vanessa and Egger, Bernhard and Palm, Christoph},
    archivePrefix={arXiv},
    eprint={2412.13244},
    year={2024}
}
```
and
```bibtex
@article{weiherer2023rbsm,
  title={Learning the shape of female breasts: an open-access 3D statistical shape model of the female breast built from 110 breast scans},
  author={Weiherer, Maximilian and Eigenberger, Andreas and Egger, Bernhard and Brébant, Vanessa and Prantl, Lukas and Palm, Christoph},
  journal={The Visual Computer},
  volume={39},
  pages={1597–-1616},
  year={2023}
}
```
Also, in case you have any questions, feel free to contact Maximilian Weiherer, Bernhard Egger, or Christoph Palm.
