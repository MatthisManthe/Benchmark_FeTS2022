import os
import json
import torch
import monai
import matplotlib.pyplot as plt
import numpy as np
from monai.config import print_config
from monai.data import DataLoader, decollate_batch, CacheDataset, LMDBDataset
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

from sklearn.cluster import AgglomerativeClustering

from data.data_utils import generate_data_dict, gen_partitioning_fets
from data.data_preprocessing import generate_train_tranform, generate_val_tranform
from plot_utils import plot_train_image_pred_output, plot_labels_outputs
from models.models import modified_get_conv_layer

variable_explorer = None

def flatten(params):
    return torch.cat([param.flatten() for param in params])
    
    
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
    clients = list(samples_inst_map_train.keys())
    print("Client order: ", clients)
    nb_clients = len(clients)
    client_to_indice = {client:i for i, client in enumerate(clients)}
    indice_to_client = {i:client for client, i in client_to_indice.items()}
    
    # Define the device to use (CPU or which GPU)
    device = torch.device(config["device"])
    transform_device = torch.device(config["tranform_device"])
    
    # Define global and local models, loss functions, optimizers and metrics  
    monai.networks.blocks.dynunet_block.get_conv_layer = modified_get_conv_layer
    cluster_models = {tuple(clients):DynUNet(
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
    ).to(device)}
    
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
    
    for (name, param) in cluster_models[tuple(clients)].named_parameters():
        print(name, param.shape)

    if config["nb_gpus"] == 1:
        cluster_models_parallel = cluster_models
        local_models_parallel = local_models
    else:
        cluster_models_parallel = {i:torch.nn.DataParallel(model) for i, model in cluster_models.items()}
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
    
    scheduler_list = {}
    for i in samples_inst_map_train.keys():
        if config["lr_scheduler"] == "plateau":
            scheduler_list[i] = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer_list[i], mode="max", factor=config["lr_factor"], patience=config["patience"], verbose=True, min_lr=config["min_lr"])
        elif config["lr_scheduler"] == "exp_decay":
            scheduler_list[i] = torch.optim.lr_scheduler.ExponentialLR(optimizer_list[i], config["gamma"], verbose=True)
        
    dice_metric_cluster = DiceMetric(include_background=True, reduction="mean", ignore_empty=False)
    dice_metric_batch_cluster = DiceMetric(include_background=True, reduction="mean_batch", ignore_empty=False)
    
    # Adding graph of model to tensorboard and print it
    writer.add_graph(cluster_models[tuple(clients)], next(iter(train_loader_list[list(samples_inst_map_train.keys())[0]]))["image"].to(device))
    print(summary(cluster_models[tuple(clients)], next(iter(train_loader_list[list(samples_inst_map_train.keys())[0]]))["image"].to(device), show_input=True, show_hierarchical=True))
    
    # Gradient storage
    updates = {client:[torch.zeros_like(param) for param in local_models[client].parameters()] for client in clients}
    
    # Initialize metrics
    best_mean_dice = {tuple(clients):-1}
    best_dice_et = {tuple(clients):-1}
    best_dice_tc = {tuple(clients):-1}
    best_dice_wt = {tuple(clients):-1}
    best_comm = {tuple(clients):0}
    
    # Training process
    for comm in range(config["max_comm_rounds"]):
        
        print("-" * 10)
        print(f"epoch {comm + 1}/{config['max_comm_rounds']}")
        
        # Communicate the cluster models to corresponding local clients
        for list_clients, model in cluster_models.items():
            for client in list_clients:
                for local_param, cluster_param in zip(local_models[client].parameters(), model.parameters()):
                    local_param.data = cluster_param.data.clone()
            
        # --------------------- First simulate each client sequentially ---------------------
        for client in samples_inst_map_train.keys():
            
            # Search for the client cluster
            cluster_group = None
            for list_clients in cluster_models.keys():
                if client in list_clients:
                    cluster_group = list_clients
                    break
                
            # Write current learning rate
            writer.add_scalar(f"Learning rate/Client {client}/Epoch", optimizer_list[client].param_groups[0]['lr'], comm+1)
            
            # Momentum restart accelerating the use of Adam locally
            if config["optim"] != "sgd" and config["momentum_restart"]:
                optimizer_list[client] = torch.optim.Adam(local_models[client].parameters(), config["learning_rate"], weight_decay=config["weight_decay"])
                  
            for local_epoch in range(config["max_local_epochs"]):
                
                # Actual number of local epochs including previous communication rounds
                epoch = comm*config['max_local_epochs'] + local_epoch
                    
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

                    loss = loss_function(outputs, labels)
                    
                    loss.backward()
                    optimizer_list[client].step()
    
                    # Get loss value on last batch
                    local_epoch_loss += loss.item()*inputs.shape[0]
                    # print(f"Client {client}, step {step}/{np.ceil(len(train_ds_list[client]) / train_loader_list[client].batch_size)}, f"train_loss: {loss.item():.4f}")
    
                # lr_scheduler.step()
                local_epoch_loss /= len(samples_inst_map_train[client])
    
                writer.add_scalar(f"Loss/train/Client {client}", local_epoch_loss, epoch+1)
                print(f"Client {client} local epoch {local_epoch + 1} average loss: {local_epoch_loss:.4f}")
        
            for id_group, (old_param, new_param) in enumerate(zip(cluster_models[cluster_group].parameters(), local_models[client].parameters())):
                updates[client][id_group] = new_param - old_param
        
        # Apply learning rate schedule step at the end of a round 
        if config["lr_scheduler"] == "exp_decay":
            for client in samples_inst_map_train.keys():
                scheduler_list[client].step()
                
        # ---------------------- Then perform server update with global evaluation -----------------------------
        
        # Compute pairwise angles
        with torch.no_grad():
            angles = torch.zeros([nb_clients, nb_clients])
            for client1 in clients:
                for client2 in clients:
                    
                    c1 = flatten(updates[client1])
                    c2 = flatten(updates[client2])
                    print(c1.shape)
                    angles[client_to_indice[client1],client_to_indice[client2]] = torch.sum(c1*c2) / (torch.norm(c1)*torch.norm(c2)+1e-12)
            
                    # Avoid memory issues
                    del c1
                    del c2
                
        angles = angles.detach().numpy()
        print("\nAngles: ", angles)
        
        current_clusters = list(cluster_models.keys())
        # Update clusters: split some clusters if needed
        for num_cluster, list_clients in enumerate(current_clusters):
            print(f"Cluster {num_cluster}, clients {list_clients}")
            
            # Compute max update norm
            max_norm = np.max([torch.norm(flatten(updates[client])).item() for client in list_clients])
            print("Max_norm: ", max_norm)
            
            # Compute mean update norm
            mean_norm = torch.norm(torch.mean(torch.stack([flatten(updates[client]) for client in list_clients]), dim=0)).item()
            print("Mean_norm: ", mean_norm)
            
            # Compute a split, either using original split condition from the paper 
            # or fixing communication rounds when a split is performed
            if (config["split_condition"] == "original" and comm > config["min_comm_round_cluster"] \
                and len(list_clients) > 2 and mean_norm < config["eps_1"] and max_norm > config["eps_2"]) \
                or (config["split_condition"] == "fixed_comms" and comm in config["clustering_comm"]):
                
                indices = [client_to_indice[client] for client in list_clients]
                
                clustering = AgglomerativeClustering(affinity="precomputed", linkage="complete").fit(-angles[indices][:,indices])

                c1 = np.argwhere(clustering.labels_ == 0).flatten() 
                c2 = np.argwhere(clustering.labels_ == 1).flatten() 
                
                print(c1, c2)
                
                list_clients1 = tuple([indice_to_client[indices[c]] for c in c1])
                list_clients2 = tuple([indice_to_client[indices[c]] for c in c2])
                
                print(list_clients1, list_clients2)
                
                # Copy the previous cluster model and informations as the basis for the two created clusters
                cluster_models[list_clients1] = copy.deepcopy(cluster_models[list_clients])
                cluster_models_parallel[list_clients1] = torch.nn.DataParallel(cluster_models[list_clients1])
                best_mean_dice[list_clients1] = -1
                best_dice_et[list_clients1] = -1
                best_dice_tc[list_clients1] = -1
                best_dice_wt[list_clients1] = -1
                best_comm[list_clients1] = -1
                
                cluster_models[list_clients2] = copy.deepcopy(cluster_models[list_clients])
                cluster_models_parallel[list_clients2] = torch.nn.DataParallel(cluster_models[list_clients2])
                best_mean_dice[list_clients2] = -1
                best_dice_et[list_clients2] = -1
                best_dice_tc[list_clients2] = -1
                best_dice_wt[list_clients2] = -1
                best_comm[list_clients2] = -1

                del cluster_models_parallel[list_clients]
                del cluster_models[list_clients]
            
                print(f"\nSplit performed, from {list_clients} to {list_clients1} and {list_clients2}")
        
        # Aggregation cluster-wise
        for list_clients in cluster_models.keys():
        
            for param in cluster_models[list_clients].parameters():
                param.data = torch.zeros_like(param.data)
            
            # Compute total number of sample in the cluster
            total_sample = 0
            for client in list_clients:
                total_sample += len(samples_inst_map_train[client])
            
            for client in list_clients:
                # Aggregate the weighted weights of the clients in global model
                for cluster_param, client_param in zip(cluster_models[list_clients].parameters(), local_models[client].parameters()):
                    cluster_param.data += client_param.data.clone() * len(samples_inst_map_train[client])/total_sample
            
        
        # Evaluation of cluster models
        if comm > config["max_comm_rounds"]*0.85 or (comm+1) % config["global_validation_interval"] == 0:
    
            for list_clients in cluster_models.keys():
                
                print(f"Evaluation of cluster {list_clients}")
                # Local metrics
                for client in list_clients:
                    
                    cluster_models[list_clients].eval()
                    
                    with torch.no_grad(): 
                        
                        for (idx, val_data) in enumerate(tqdm(val_loader_list[client])):
                            
                            val_inputs, val_labels = (
                                val_data["image"].to(device),
                                val_data["label"].to(device),
                            )
                            
                            # Sliding window inference and thresholding
                            val_outputs = sliding_window_inference(val_inputs, config["roi_size"], batch_size, cluster_models_parallel[list_clients], overlap=config["sliding_window_overlap"], mode="gaussian")
                            val_outputs = [post_trans(i) for i in decollate_batch(val_outputs)]
        
                            # compute metric for current iteration
                            dice_metric_cluster(y_pred=val_outputs, y=val_labels)
                            dice_metric_batch_cluster(y_pred=val_outputs, y=val_labels)
            
                # aggregate the final mean dice result
                metric_d = dice_metric_cluster.aggregate().item()
        
                # Aggregate the final mean dices for each label
                metric_batch = dice_metric_batch_cluster.aggregate()
                metric_tc = metric_batch[0].item()
                metric_wt = metric_batch[1].item()
                metric_et = metric_batch[2].item()
        
                # Reset metrics objects
                dice_metric_cluster.reset()
                dice_metric_batch_cluster.reset()
        
                writer.add_scalars("Validation/Cluster models/Mean_dice", {str(list_clients):metric_d}, comm+1)
                writer.add_scalars("Validation/Cluster models/Tumor core dice", {str(list_clients):metric_tc}, comm+1)
                writer.add_scalars("Validation/Cluster models/Whole tumor dice", {str(list_clients):metric_wt}, comm+1)
                writer.add_scalars("Validation/Cluster models/Enhancing tumor dice", {str(list_clients):metric_et}, comm+1)
        
                if metric_d > best_mean_dice[list_clients]:
                    best_mean_dice[list_clients] = metric_d
                    best_dice_et[list_clients] = metric_et
                    best_dice_tc[list_clients] = metric_tc
                    best_dice_wt[list_clients] = metric_wt
                    best_comm[list_clients] = comm + 1
                    torch.save(cluster_models[list_clients].state_dict(), os.path.join(log_dir, f"best_cluster_model_{str(list_clients)}.pth"))
                    print("\nSaved new best metric global model")
                print(
                    f"\nCurrent epoch: {comm + 1}, current mean dice: {metric_d:.4f}"
                    f"\nmean dice tc: {metric_tc:.4f}"
                    f"\nmean dice wt: {metric_wt:.4f}"
                    f"\nmean dice et: {metric_et:.4f}"
                    f"\nbest mean dice: {best_mean_dice[list_clients]:.4f} "
                    f"at global epoch: {best_comm[list_clients]}"
                )
                
    result_dict = {"hparam/Mean_dice":0,
                    "hparam/Dice_ET":0,
                    "hparam/Dice_TC":0,
                    "hparam/Dice_WT":0}
    num_val_samples = {}
    for list_clients in cluster_models.keys():
        result_dict[f"hparam/Mean_dice/{str(list_clients)}"] = best_mean_dice[list_clients]
        result_dict[f"hparam/Dice_ET/{str(list_clients)}"] = best_dice_et[list_clients]
        result_dict[f"hparam/Dice_TC/{str(list_clients)}"] = best_dice_tc[list_clients]
        result_dict[f"hparam/Dice_WT/{str(list_clients)}"] = best_dice_wt[list_clients]
        result_dict[f"hparam/Comm_round/{str(list_clients)}"] = best_comm[list_clients]
        
        total_sample = 0
        for client in list_clients:
            total_sample += len(samples_inst_map_train[client])
            
        result_dict["hparam/Mean_dice"] += best_mean_dice[list_clients] * total_sample / len(val_dict)
        result_dict["hparam/Dice_ET"] += best_dice_et[list_clients] * total_sample / len(val_dict)
        result_dict["hparam/Dice_TC"] += best_dice_tc[list_clients] * total_sample / len(val_dict)
        result_dict["hparam/Dice_WT"] += best_dice_wt[list_clients] * total_sample / len(val_dict)
    
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
