import os
import json
import torch
import monai
import matplotlib.pyplot as plt
from monai.config import print_config
from monai.data import DataLoader, decollate_batch, CacheDataset, LMDBDataset, Dataset
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

from data.data_utils import generate_data_dict, gen_partitioning_fets, DataLoaderWithMemory
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
    
    for client in samples_inst_map_train.keys():
        print("Client ", client, "train size: ", len(samples_inst_map_train[client]))
        
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
    global_model = DynUNet(
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
    ).to(device)
    
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
    
    for (name, param) in global_model.named_parameters():
        print(name, param.shape)

    if config["nb_gpus"] == 1:
        global_model_parallel = global_model
        local_models_parallel = local_models
    else:
        global_model_parallel = torch.nn.DataParallel(global_model)
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
        train_loader_list[i] = DataLoaderWithMemory(DataLoader(train_ds_list[i], batch_size=batch_size, shuffle=True, num_workers=0))
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
        
    dice_metric = DiceMetric(include_background=True, reduction="mean", ignore_empty=False)
    dice_metric_batch = DiceMetric(include_background=True, reduction="mean_batch", ignore_empty=False)
    
    # Adding graph of model to tensorboard and print it
    writer.add_graph(global_model, train_loader_list[list(samples_inst_map_train.keys())[0]].get_samples()["image"].to(device))
    print(summary(global_model, train_loader_list[list(samples_inst_map_train.keys())[0]].get_samples()["image"].to(device), show_input=True, show_hierarchical=True))
    
    # Initialize metrics
    best_mean_dice = -1
    best_dice_et = -1
    best_dice_tc = -1
    best_dice_wt = -1
    best_comm = 0
    
    # Training process
    for comm in range(config["max_comm_rounds"]):
        
        print("-" * 10)
        print(f"epoch {comm + 1}/{config['max_comm_rounds']}")
        
        
        # Communicate the global model to every local clients
        if config["norm"] != "BATCH":
            for client in samples_inst_map_train.keys():
                for local_param, global_param in zip(local_models[client].parameters(), global_model.parameters()):
                    local_param.data = global_param.data.clone()
            
        # --------------------- First simulate each client sequentially ---------------------
        for client in samples_inst_map_train.keys():
            
            # Write current learning rate
            writer.add_scalar(f"Learning rate/Client {client}/Epoch", optimizer_list[client].param_groups[0]['lr'], comm+1)
            
            # Momentum restart accelerating the use of Adam locally
            if config["optim"] != "sgd" and config["momentum_restart"]:
                optimizer_list[client] = torch.optim.Adam(local_models[client].parameters(), config["learning_rate"], weight_decay=config["weight_decay"])
                  
            local_models[client].train()
            
            local_epoch_loss = 0
            nb_sample_seen = 0
            
            # Perform a fixed amount of updates (not epoch)
            for local_iteration in tqdm(range(config["max_local_iterations"])):
                
                # Actual number of local epochs including previous communication rounds
                local_epoch = comm*config['max_local_iterations'] + local_iteration
                    
                # ----------- Local training step -------------  
                # Get one batch 
                batch_data = train_loader_list[client].get_samples()
                
                inputs, labels = (
                    batch_data["image"].to(device),
                    batch_data["label"].to(device),
                )

                optimizer_list[client].zero_grad()
                outputs = local_models_parallel[client](inputs)
                    
                # Evaluate loss, backward and step from optimizer, with or without local prox regularization
                if config["prox"]:
                    proximal_term = 0.0
                    for w, w_t in zip(local_models[client].parameters(), global_model.parameters()):
                        proximal_term += (w - w_t).norm(2)
        
                    loss = loss_function(outputs, labels) + (config["mu"] / 2.0) * proximal_term
                    
                else:
                    loss = loss_function(outputs, labels)
                
                loss.backward()
                optimizer_list[client].step()

                # Get loss value on last batch
                local_epoch_loss += loss.item()*inputs.shape[0]
                nb_sample_seen += inputs.shape[0]
                # print(f"Client {client}, step {step}/{np.ceil(len(train_ds_list[client]) / train_loader_list[client].batch_size)}, f"train_loss: {loss.item():.4f}")
    
                writer.add_scalar(f"Loss/running_train/Client {client}", loss.item(), local_epoch+1)
                print(f"Client {client} local epoch {local_epoch + 1} average loss: {loss.item():.4f}")
                
            # lr_scheduler.step()
            local_epoch_loss /= nb_sample_seen

            writer.add_scalar(f"Loss/train/Client {client}", local_epoch_loss, comm+1)
            print(f"Client {client} end of local training average loss: {local_epoch_loss:.4f}")
                        
        # Apply learning rate schedule step at the end of a round 
        if config["lr_scheduler"] == "exp_decay":
            for client in samples_inst_map_train.keys():
                scheduler_list[client].step()
                
        # ---------------------- Then perform server update with global evaluation -----------------------------
        
        # First try with simple FedAvg aggregation method: compute the average
        # Set global model parameters to 0
        
        # ---------------- Batch norm version not up to date ----------------
        if config["norm"] == "BATCH":
            for key in global_model.state_dict().keys():
                # num_batches_tracked is a non trainable LongTensor and
                # num_batches_tracked are the same for all clients for the given datasets
                if 'num_batches_tracked' in key:
                    global_model.state_dict()[key].data.copy_(local_models[0].state_dict()[key])
                else:
                    # Average parameters (and buffers ...)
                    temp = torch.zeros_like(global_model.state_dict()[key])
                    for client in range(nb_clients):
                        temp += local_models[client].state_dict()[key] * len(samples_inst_map_train[client])/len(train_dict)
                    global_model.state_dict()[key].data.copy_(temp)
                    # Communicate new state to each client
                    for client in range(nb_clients):
                        local_models[client].state_dict()[key].data.copy_(global_model.state_dict()[key])
        else:      
            for param in global_model.parameters():
                param.data = torch.zeros_like(param.data)
            
            for client in samples_inst_map_train.keys():  
                # Aggregate the weighted weights of the clients in global model
                for global_param, client_param in zip(global_model.parameters(), local_models[client].parameters()):
                    global_param.data += client_param.data.clone() * len(samples_inst_map_train[client])/len(train_dict)
            
        
        # Evaluation of global model
        if comm > config["max_comm_rounds"]*0.85 or (comm+1) % config["global_validation_interval"] == 0:
    
            # Local metrics
            for client in samples_inst_map_train.keys():
                
                global_model.eval()
                
                with torch.no_grad(): 
                    
                    for (idx, val_data) in enumerate(tqdm(val_loader_list[client])):
                        
                        val_inputs, val_labels = (
                            val_data["image"].to(device),
                            val_data["label"].to(device),
                        )
                        
                        # Sliding window inference and thresholding
                        val_outputs = sliding_window_inference(val_inputs, config["roi_size"], batch_size, global_model_parallel, overlap=config["sliding_window_overlap"], mode="gaussian")
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
    
            writer.add_scalar("Validation/Global model/Mean_dice", metric_d, comm+1)
            writer.add_scalar("Validation/Global model/Tumor core dice", metric_tc, comm+1)
            writer.add_scalar("Validation/Global model/Whole tumor dice", metric_wt, comm+1)
            writer.add_scalar("Validation/Global model/Enhancing tumor dice", metric_et, comm+1)
    
            if metric_d > best_mean_dice:
                best_mean_dice = metric_d
                best_dice_et = metric_et
                best_dice_tc = metric_tc
                best_dice_wt = metric_wt
                best_comm = comm + 1
                torch.save(global_model.state_dict(), os.path.join(log_dir, "best_global_model.pth"))
                print("\nSaved new best metric global model")
            print(
                f"\nCurrent epoch: {comm + 1}, current mean dice: {metric_d:.4f}"
                f"\nmean dice tc: {metric_tc:.4f}"
                f"\nmean dice wt: {metric_wt:.4f}"
                f"\nmean dice et: {metric_et:.4f}"
                f"\nbest mean dice: {best_mean_dice:.4f} "
                f"at global epoch: {best_comm}"
            )
            
            if (comm + 1) in config["save_comms"]:
                shutil.copyfile(os.path.join(log_dir, "best_global_model.pth"), os.path.join(log_dir, f"best_global_model_comm_{comm+1}.pth"))
                
    # Adding hyperparameters value to tensorboard
    config_hparam = {}
    for key, value in config.items():
        if type(value) is list:
            value = torch.Tensor(value)
        config_hparam[key] = value
    writer.add_hparams(config_hparam, {"hparam/Mean_dice":best_mean_dice,
                                       "hparam/Dice_ET":best_dice_et,
                                       "hparam/Dice_TC":best_dice_tc,
                                       "hparam/Dice_WT":best_dice_wt,
                                       "hparam/Comm_round":best_comm})    
                    


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
