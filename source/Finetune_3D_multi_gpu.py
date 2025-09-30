import os
import json
import torch
import numpy as np
import monai
import matplotlib.pyplot as plt
from monai.config import print_config
from monai.data import DataLoader, decollate_batch, CacheDataset, Dataset, LMDBDataset
from monai.losses import DiceLoss
from monai.metrics import DiceMetric
from monai.inferers import sliding_window_inference
from monai.networks.nets import DynUNet
from monai.transforms import (
    Activations,
    AsDiscrete,
    EnsureType,
    Compose
)
import glob
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter
from pytorch_model_summary import summary
from time import gmtime, strftime
import argparse
import shutil
import copy

from data.data_utils import generate_data_dict, gen_partitioning_fets
from data.data_preprocessing import generate_train_tranform, generate_val_tranform
from plot_utils import plot_train_image_pred_output, plot_labels_outputs
from models.models import modified_get_conv_layer

variable_explorer = None

def main(args, config):
    """Main function"""
    
    # Initializing tensorboard summary
    log_dir = config["log_dir"] + "/" + os.path.basename(__file__)
    if config["log_filename"] is not None:
        log_dir += "/"+config["log_filename"]
    else:
        log_dir += "/"+strftime("%Y-%m-%d_%H-%M-%S", gmtime())
    writer = SummaryWriter(log_dir=log_dir)
    
    # Copy config file in the experiment folder
    shutil.copy(args.config_path, log_dir)
    
    # Generating list of data files to load
    data_dir = config['data_dir']
    data_dict = generate_data_dict(data_dir)
    
    # Loading training and validation sets
    train_part_list, val_part_list, train_dict, val_dict, \
        samples_inst_map_train, samples_inst_map_val = gen_partitioning_fets(data_dict, 
                                                                                config["partition_file"], 
                                                                                prop_full_dataset=config['prop_full_dataset'], 
                                                                                ratio_train=config['ratio_train'])
    # Save the partitioning training / validation in the experiment folder
    with open(os.path.join(log_dir,'training_samples.json'), 'w') as fp:
        json.dump(samples_inst_map_train, fp, indent=4)
        
    with open(os.path.join(log_dir,'validation_samples.json'), 'w') as fp:
        json.dump(samples_inst_map_val, fp, indent=4)
    
    # Define the number of clients in the federation
    nb_clients = len(train_part_list)
    
    # Define the device to use (CPU or which GPU)
    device = torch.device(config["device"])
    transform_device = torch.device(config["tranform_device"])
    
    # Define global and local models, loss functions, optimizers and metrics  
    monai.networks.blocks.dynunet_block.get_conv_layer = modified_get_conv_layer 
    local_models = {i:DynUNet(
        spatial_dims = 3,
        in_channels = 4,
        out_channels = 3,
        kernel_size = config["kernel_sizes"],
        filters = config["filters"],
        strides = config["strides"],
        upsample_kernel_size = config["strides"][1:],
        norm_name=(config["norm"], {"affine": config["learn_affine_norm"]} if config["norm"] != "GROUP" else {"affine":config["learn_affine_norm"], "num_groups":1}),
        act_name=("LeakyReLu", {"negative_slope":0.01}),
        trans_bias=True,
    ).to(device) for i in samples_inst_map_train.keys()}
    
    # Initialize the local model as the best obtained federated model.
    for client in samples_inst_map_train.keys():
        #local_models[client] = torch.load(os.path.join(config["model_dir"], config["model_file"]))
        local_models[client].load_state_dict(torch.load(os.path.join(config["model_dir"], config["model_file"])))
    
    for (name, param) in local_models["1"].named_parameters():
        print(name, param.shape)

    if config["nb_gpus"] == 1:
        local_models_parallel = local_models
    else:
        local_models_parallel = {i:torch.nn.DataParallel(model) for i, model in local_models.items()}
        
    # Define the transformations to be applied to data
    train_transform = generate_train_tranform(roi_size=config["roi_size"], data_aug=config["data_augmentation"], device=transform_device)
    val_transform = generate_val_tranform(roi_size=config["roi_size"])
    post_trans = Compose(
        [EnsureType(), Activations(sigmoid=True), AsDiscrete(threshold_values=True)]
    )

    # Define the batchsize
    batch_size = config["nb_gpus"]*config["batch_size_per_gpu"]
    
    # Initialize lists of train and validation datasets and loaders
    train_ds_list = {}
    train_loader_list = {}
    val_ds_list = {}
    val_loader_list = {}
    
    # Define train and validation datasets and dataloaders for each
    if config["persistent_dataset"]:
        # Delete every cached elements
        cache_dir = config["cache_dir"]
        filelist = glob.glob(os.path.join(cache_dir, "*"))
        for f in filelist:
            os.remove(f)
            
        for i in samples_inst_map_train.keys():
            train_ds_list[i] = LMDBDataset(data=samples_inst_map_train[i], transform=train_transform, cache_dir=config["cache_dir"])
            val_ds_list[i] = LMDBDataset(data=samples_inst_map_val[i], transform=val_transform, cache_dir=config["cache_dir"])
    else:
        for i in samples_inst_map_train.keys():
            train_ds_list[i] = CacheDataset(data=samples_inst_map_train[i], transform=train_transform)
            val_ds_list[i] = CacheDataset(data=samples_inst_map_val[i], transform=val_transform)
    
    for i in samples_inst_map_train.keys():
        train_loader_list[i] = DataLoader(train_ds_list[i], batch_size=batch_size, shuffle=True, num_workers=0)
        val_loader_list[i] = DataLoader(val_ds_list[i], batch_size=1, shuffle=False, num_workers=0)
    
    # Define loss function, optimizer and metrics
    loss_function = DiceLoss(sigmoid=True, smooth_nr=1, smooth_dr=1, squared_pred=False)
    
    optimizer_list = None
    if config["optim"] == "sgd":
        optimizer_list = {i:torch.optim.SGD(local_models[i].parameters(), config["learning_rate"], weight_decay=config["weight_decay"]) for i in samples_inst_map_train.keys()}
    else:
        optimizer_list = {i:torch.optim.Adam(local_models[i].parameters(), config["learning_rate"], weight_decay=config["weight_decay"]) for i in samples_inst_map_train.keys()}
        
    dice_metric = DiceMetric(include_background=True, reduction="mean", ignore_empty=False)
    dice_metric_batch = DiceMetric(include_background=True, reduction="mean_batch", ignore_empty=False)
    
    # Adding graph of model to tensorboard and print it
    writer.add_graph(local_models[list(samples_inst_map_train.keys())[0]], next(iter(train_loader_list[list(samples_inst_map_train.keys())[0]]))["image"].to(device))
    print(summary(local_models[list(samples_inst_map_train.keys())[0]], next(iter(train_loader_list[list(samples_inst_map_train.keys())[0]]))["image"].to(device), show_input=True, show_hierarchical=True))
        
    result_dict = {}
    best_mean_dices = {}
    best_dices_et = {}
    best_dices_tc = {}
    best_dices_wt = {}

    # --------------------- First simulate each client sequentially ---------------------
    for client in samples_inst_map_train.keys():
        
        # Initialize metrics
        best_mean_dice = -1
        best_dice_et = -1
        best_dice_tc = -1
        best_dice_wt = -1
        best_local_epoch = 0
        
        # get validation result from the model without finetuning
        local_models[client].eval()
        
        with torch.no_grad():
            
            for (idx, val_data) in enumerate(tqdm(val_loader_list[client])):
                
                # Put data batch in GPU
                val_inputs, val_labels = (
                    val_data["image"].to(device),
                    val_data["label"].to(device),
                )
                
                # Sliding window inference keeping the same roi size as in training
                val_outputs = sliding_window_inference(val_inputs, config["roi_size"], batch_size, local_models_parallel[client], overlap=config["sliding_window_overlap"], mode="gaussian")
                
                # Thresholding at 0.5 to get binary segmentation map
                val_outputs = [post_trans(i) for i in decollate_batch(val_outputs)]

                # compute metric for current iteration
                dice_metric(y_pred=val_outputs, y=val_labels)
                dice_metric_batch(y_pred=val_outputs, y=val_labels)

            # aggregate the final mean dice result
            metric_d = dice_metric.aggregate().item()

            # Aggregate the final mean dices for each label
            metric_batch = dice_metric_batch.aggregate()
            metric_tc = metric_batch[0].item()
            metric_wt = metric_batch[1].item()
            metric_et = metric_batch[2].item()

            # Reset metrics objects
            dice_metric.reset()
            dice_metric_batch.reset()
            
            writer.add_scalar(f"Validation/Client {client}/Mean_dice", metric_d, 0)
            writer.add_scalar(f"Validation/Client {client}/Tumor core dice", metric_tc, 0)
            writer.add_scalar(f"Validation/Client {client}/Whole tumor dice", metric_wt, 0)
            writer.add_scalar(f"Validation/Client {client}/Enhancing tumor dice", metric_et, 0)

            if metric_d > best_mean_dice:
                best_mean_dice = metric_d
                best_dice_et = metric_et
                best_dice_tc = metric_tc
                best_dice_wt = metric_wt
                best_local_epoch = 0
                torch.save(local_models[client].state_dict(), os.path.join(log_dir, f"best_model_client_{client}.pth"))
                print(f"\nSaved new best metric local model for client {client}")
            print(
                f"\nCurrent epoch: 0, current mean dice: {metric_d:.4f}"
                f"\nmean dice tc: {metric_tc:.4f}"
                f"\nmean dice wt: {metric_wt:.4f}"
                f"\nmean dice et: {metric_et:.4f}"
                f"\nbest mean dice: {best_mean_dice:.4f} "
                f"at global epoch: {best_local_epoch}"
            )
        
        # Finetune the model for some epochs
        for local_epoch in range(config["max_local_epochs"]):
                
            # ----------- Local training step -------------
            
            local_models[client].train()
            
            local_epoch_loss = 0
            step = 0
    
            for (idx, batch_data) in enumerate(tqdm(train_loader_list[client])):
                step += 1
                inputs, labels = (
                    batch_data["image"].to(device),
                    batch_data["label"].to(device),
                )
    
                optimizer_list[client].zero_grad()
                outputs = local_models_parallel[client](inputs)
                    
                # Evaluate loss, backward and step from optimizer
                loss = loss_function(outputs, labels)
                
                loss.backward()
                optimizer_list[client].step()
    
                # Get loss value on last batch
                local_epoch_loss += loss.item()*inputs.shape[0]
                # print(f"Client {client}, step {step}/{np.ceil(len(train_ds_list[client]) / train_loader_list[client].batch_size)}, f"train_loss: {loss.item():.4f}")
    
            # lr_scheduler.step()
            local_epoch_loss /= len(samples_inst_map_train[client])
    
            writer.add_scalar(f"Loss/train/Client {client}", local_epoch_loss, local_epoch+1)
            print(f"Client {client} local epoch {local_epoch + 1} average loss: {local_epoch_loss:.4f}")
            
            # --------------- Local validation step ---------------
            if (local_epoch) % config['local_validation_interval'] == 0:
                
                local_models[client].eval()
                
                with torch.no_grad():
                    
                    for (idx, val_data) in enumerate(tqdm(val_loader_list[client])):
                        
                        # Put data batch in GPU
                        val_inputs, val_labels = (
                            val_data["image"].to(device),
                            val_data["label"].to(device),
                        )
                        
                        # Sliding window inference keeping the same roi size as in training
                        val_outputs = sliding_window_inference(val_inputs, config["roi_size"], batch_size, local_models_parallel[client], overlap=config["sliding_window_overlap"], mode="gaussian")
                        
                        # Thresholding at 0.5 to get binary segmentation map
                        val_outputs = [post_trans(i) for i in decollate_batch(val_outputs)]
    
                        # compute metric for current iteration
                        dice_metric(y_pred=val_outputs, y=val_labels)
                        dice_metric_batch(y_pred=val_outputs, y=val_labels)
    
                    # aggregate the final mean dice result
                    metric_d = dice_metric.aggregate().item()
    
                    # Aggregate the final mean dices for each label
                    metric_batch = dice_metric_batch.aggregate()
                    metric_tc = metric_batch[0].item()
                    metric_wt = metric_batch[1].item()
                    metric_et = metric_batch[2].item()
    
                    # Reset metrics objects
                    dice_metric.reset()
                    dice_metric_batch.reset()
                    
                    writer.add_scalar(f"Validation/Client {client}/Mean_dice", metric_d, local_epoch+1)
                    writer.add_scalar(f"Validation/Client {client}/Tumor core dice", metric_tc, local_epoch+1)
                    writer.add_scalar(f"Validation/Client {client}/Whole tumor dice", metric_wt, local_epoch+1)
                    writer.add_scalar(f"Validation/Client {client}/Enhancing tumor dice", metric_et, local_epoch+1)
        
                    if metric_d > best_mean_dice:
                        best_mean_dice = metric_d
                        best_dice_et = metric_et
                        best_dice_tc = metric_tc
                        best_dice_wt = metric_wt
                        best_local_epoch = local_epoch + 1
                        torch.save(local_models[client].state_dict(), os.path.join(log_dir, f"best_model_client_{client}.pth"))
                        print(f"\nSaved new best metric local model for client {client}")
                    print(
                        f"\nCurrent epoch: {local_epoch + 1}, current mean dice: {metric_d:.4f}"
                        f"\nmean dice tc: {metric_tc:.4f}"
                        f"\nmean dice wt: {metric_wt:.4f}"
                        f"\nmean dice et: {metric_et:.4f}"
                        f"\nbest mean dice: {best_mean_dice:.4f} "
                        f"at global epoch: {best_local_epoch}"
                    )
            
        result_dict[f"hparam/Mean_dice/client_{client}"] = best_mean_dice
        result_dict[f"hparam/Dice_ET/client_{client}"] = best_dice_et
        result_dict[f"hparam/Dice_TC/client_{client}"] = best_dice_tc
        result_dict[f"hparam/Dice_WT/client_{client}"] = best_dice_wt
        
        best_mean_dices[client] = best_mean_dice
        best_dices_et[client] = best_dice_et
        best_dices_tc[client] = best_dice_tc
        best_dices_wt[client] = best_dice_wt
    
    result_dict["hparam/Mean_dice"] = np.sum([best_mean_dices[client]*len(samples_inst_map_val[client]) for client in samples_inst_map_train.keys()])/len(val_dict)
    result_dict["hparam/Dice_ET"] = np.sum([best_dices_et[client]*len(samples_inst_map_val[client]) for client in samples_inst_map_train.keys()])/len(val_dict)
    result_dict["hparam/Dice_TC"] = np.sum([best_dices_tc[client]*len(samples_inst_map_val[client]) for client in samples_inst_map_train.keys()])/len(val_dict)
    result_dict["hparam/Dice_WT"] = np.sum([best_dices_wt[client]*len(samples_inst_map_val[client]) for client in samples_inst_map_train.keys()])/len(val_dict)
    
    # Adding hyperparameters value to tensorboard
    config_hparam = {}
    for key, value in config.items():
        if type(value) is list:
            value = torch.Tensor(value)
        config_hparam[key] = value
    writer.add_hparams(config_hparam, result_dict)    
                    


if __name__ == "__main__":
    
    print("Curent working directory: ", os.getcwd())
    
    print("Is cuda avaiable? ", torch.cuda.is_available())
    print("Number of cuda devices available: ", torch.cuda.device_count())
    
    print("Monai config")
    print_config()
    
    # Define argument parser and its attributes
    parser = argparse.ArgumentParser(description='Train 3D UNet on Brats')
    
    parser.add_argument('--config_path', dest='config_path', type=str,
                        help='path to json config file')
    
    # Parse the arguments
    args = parser.parse_args()
    
    # Read the config file
    with open(args.config_path, 'r') as f:
        config = json.load(f)
    
    main(args, config)
