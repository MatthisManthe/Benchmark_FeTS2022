# Federated brain tumor segmentation: an extensive benchmark

This is the official code base for the paper 

> [Matthis Manthe et al., “Federated Brain Tumor Segmentation: An Extensive Benchmark,” Medical Image Analysis 97 (October 2024)](https://doi.org/10.1016/j.media.2024.103270)

## Code base structure
The structure of the code base is simple
- Directly in the ```/source``` folder is one python code for each training algorithm tested in the paper: *Centralized*, *Local training on institution 1*, *FedAvg* (with fixed local epochs and local iterations), *FedAvg IID*, *FedAdam*, *FedNova*, *SCAFFOLD*, *q-FedAvg*, *FedPIDAvg*, *Local finetuning*, *Ditto*, *FedPer* (accounting for *FedPer* and *LG-FedAvg*), *CFL* and *prior clustered FedAvg*,
- In the directory ```source/config``` can be found one json config file for each training algorithm necessary to start an experiment. They are copies of each config file selected through grid searches, used to generate the final results of our paper (for one fold). Note that the hyperparameters for personalized methods were selected per institution, requiring the full grid search to replicate the results of the paper.
- In the folder ```/source/test``` are the testing codes producing final performances based on the output of training algorithms (with associated ```/source/test/config``` folder with json files).

The fold splits of FeTS2022 dataset used in cross-validations can be found in the folder ```/data_folds```.

## Launching an experiment
All these python files can be ran using the following typical command

```python3 NAME.py --config_path config/CONFIG_NAME.json```

which, for FedAvg using the config file for the FeTS2022 challenge partitioning defined in ```config_FedAvg.json```, becomes 

```python3 Train_FedAvg_3D_multi_gpu.py --config_path config/config_FedAvg.json```

One needs to create a ```/runs``` directory for experiment folders to be created every time a code is ran, containing everything related to the experiment instance (tensorboard, model weights, copy of the config file, figures, etc.).

## Dependencies
The main frameworks used are essentially 
- Pytorch
- Numpy
- Sklearn
- Monai

with additional dependencies with tqdm, glob, pandas, pickle and pacmap.

## Multi-GPU implementation
Note that all of these implementations can utilize multi-GPU nodes. As we used a 3D-UNet without problematic layers from a distributed computation point of view (such as batch norm), and model copy operations being negligible in cost compared to forward passes in our case, we leveraged the simple ```torch.nn.DataParallel``` module to achieve GPU parallelization of batches.

