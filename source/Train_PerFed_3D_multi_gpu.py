import os
import json
import torch
import matplotlib.pyplot as plt
import monai
import numpy as np
from monai.config import print_config
from monai.data import DataLoader, decollate_batch, Dataset, CacheDataset, LMDBDataset
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
    
    # Define the number of clients in the federation
    nb_clients = len(train_part_list)
    
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
 
    print("\nGlobal model parameter groups")
    for (name, param) in global_model.named_parameters():
        print(name, param.shape)

    if config["nb_gpus"] == 1:
        global_model_parallel = global_model
        local_models_parallel = local_models
    else:
        global_model_parallel = torch.nn.DataParallel(global_model)
        local_models_parallel = {i:torch.nn.DataParallel(model) for i, model in local_models.items()}
           
    # Get total number of parameter groups
    total_nb_param_group = len(list(global_model.parameters()))
    param_groups = [(name,param.shape) for name, param in global_model.named_parameters()]
    
    print(config["nb_local_blocks"], total_nb_param_group, slice(total_nb_param_group - config["nb_local_blocks"], total_nb_param_group))
    
    # Get the indices of the parameter groups to share by FedAvg
    if config["starting_block"] == "first":
        index_shared_param_groups = slice(config["nb_local_blocks"], total_nb_param_group)
    elif config["starting_block"] == "last":
        index_shared_param_groups = slice(0, total_nb_param_group - config["nb_local_blocks"])
    elif config["starting_block"] == "first_and_last":
        index_shared_param_groups = slice(config["nb_local_blocks_first"], total_nb_param_group - config["nb_local_blocks_last"])
        
    print("\nSupposed functioning:")
    for idx, (name, param) in enumerate(global_model.named_parameters()):
        if (config["starting_block"] == "first" and idx < config["nb_local_blocks"]) or \
            (config["starting_block"] == "last" and idx >= len(list(global_model.parameters())) - config["nb_local_blocks"]) or \
            (config["starting_block"] == "first_and_last" and idx < config["nb_local_blocks_first"] or idx >= len(list(global_model.parameters())) - config["nb_local_blocks_last"]):
            print("Local block: ", name, param.shape)
        else:
            print("Global block: ", name, param.shape)
           
    print("\nSlicing: ")
    print(index_shared_param_groups)
    print(param_groups[index_shared_param_groups])
    
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
            
    dice_metric = DiceMetric(include_background=True, reduction="mean", ignore_empty=False)
    dice_metric_batch = DiceMetric(include_background=True, reduction="mean_batch", ignore_empty=False)
    dice_metric_participant = DiceMetric(include_background=True, reduction="mean", ignore_empty=False)
    dice_metric_batch_participant = DiceMetric(include_background=True, reduction="mean_batch", ignore_empty=False)
    
    
    # Adding graph of model to tensorboard and print it
    writer.add_graph(global_model, next(iter(train_loader_list[list(samples_inst_map_train.keys())[0]]))["image"].to(device))
    print(summary(global_model, next(iter(train_loader_list[list(samples_inst_map_train.keys())[0]]))["image"].to(device), show_input=True, show_hierarchical=True))
    
    # Initialize metrics
    best_mean_dice = {client:-1 for client in samples_inst_map_train.keys()}
    best_dice_et = {client:-1 for client in samples_inst_map_train.keys()}
    best_dice_tc = {client:-1 for client in samples_inst_map_train.keys()}
    best_dice_wt = {client:-1 for client in samples_inst_map_train.keys()}
    best_comm = {client:0 for client in samples_inst_map_train.keys()}
    
    # Every participants start with the same initialization
    for client in samples_inst_map_train.keys():
        for local_param, global_param in zip(local_models[client].parameters(), global_model.parameters()):
            local_param.data = global_param.data.clone()
            
    # Training process
    for comm in range(config["max_comm_rounds"]):
        
        print("-" * 10)
        print(f"epoch {comm + 1}/{config['max_comm_rounds']}")
            
        # --------------------- First simulate each client sequentially ---------------------
        for client in samples_inst_map_train.keys():
                
            # Write current learning rate
            writer.add_scalar(f"Learning rate/Client {client}/Epoch", optimizer_list[client].param_groups[0]['lr'], comm+1)
            
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
                    
                    # Plot predictions
                    if comm==0 and local_epoch==0 and idx<4 and False:
                        with torch.no_grad():
                            fig = plot_train_image_pred_output(inputs, outputs, labels)
                            writer.add_figure(f"Initial check/Client {client}", fig, comm+1, close=False)
                            plt.show()
                        
                    # Evaluate loss, backward and step from optimizer
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
                
                # --------------- Local validation step ---------------
                if False and (local_epoch) % config['local_validation_interval'] == 0:
                    
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
    
                            # Show some images if needed
                            if (epoch + 1) % config['local_print_validation_interval'] == 0 and False:
                                fig = plot_labels_outputs(val_inputs, val_outputs, val_labels)
                                writer.add_figure(f"Validation plot/Client {client}/{epoch+1}", fig, epoch+1, close=False)
                                plt.show()
    
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
    
                        writer.add_scalar(f"Validation/Client {client}/Mean_dice", metric_d, epoch+1)
                        writer.add_scalar(f"Validation/Client {client}/Tumor core dice", metric_tc, epoch+1)
                        writer.add_scalar(f"Validation/Client {client}/Whole tumor dice", metric_wt, epoch+1)
                        writer.add_scalar(f"Validation/Client {client}/Enhancing tumor dice", metric_et, epoch+1)

        # Apply learning rate schedule step at the end of a round 
        if config["lr_scheduler"] == "exp_decay":
            for client in samples_inst_map_train.keys():
                scheduler_list[client].step()

        # --------------Then perform server update, communication, and evaluation -------------------
        
        # Set global model parameters to 0
        for param in global_model.parameters():
            param.data = torch.zeros_like(param.data)
                
        for client in samples_inst_map_train.keys():  
            # Aggregate the weighted weights of the clients in global model
            for global_param, client_param in zip(list(global_model.parameters())[index_shared_param_groups], list(local_models[client].parameters())[index_shared_param_groups]):
                global_param.data += client_param.data.clone() * len(samples_inst_map_train[client])/len(train_dict)
                
        # Communicate the global layers to every local clients
        for client in samples_inst_map_train.keys():
            for local_param, global_param in zip(list(local_models[client].parameters())[index_shared_param_groups], list(global_model.parameters())[index_shared_param_groups]):
                local_param.data = global_param.data.clone()
                
        # Then perform a validation step using personalized models
        if comm % config["global_validation_interval"] == 0:
    
            # Local metrics
            for client in samples_inst_map_train.keys():
                
                local_models[client].eval()
                
                with torch.no_grad(): 
                    
                    for (idx, val_data) in enumerate(tqdm(val_loader_list[client])):
                        
                        val_inputs, val_labels = (
                            val_data["image"].to(device),
                            val_data["label"].to(device),
                        )
                        
                        # Sliding window inference and thresholding
                        val_outputs = sliding_window_inference(val_inputs, config["roi_size"], batch_size, local_models_parallel[client], overlap=config["sliding_window_overlap"], mode="gaussian")
                        val_outputs = [post_trans(i) for i in decollate_batch(val_outputs)]
    
                        # Plot predictions on evaluation set
                        if (comm + 1) % config["global_print_validation_interval"] == 0 and False:
                            fig = plot_labels_outputs(val_inputs, val_outputs, val_labels)
                            writer.add_figure(f"Validation plot/Global model/{epoch+1}", fig, comm+1, close=False)
                            plt.show()
    
                        # compute metric for current iteration
                        dice_metric(y_pred=val_outputs, y=val_labels)
                        dice_metric_batch(y_pred=val_outputs, y=val_labels)
                        dice_metric_participant(y_pred=val_outputs, y=val_labels)
                        dice_metric_batch_participant(y_pred=val_outputs, y=val_labels)
            
                # aggregate the final mean dice result
                metric_d = dice_metric_participant.aggregate().item()
        
                # Aggregate the final mean dices for each label
                metric_batch = dice_metric_batch_participant.aggregate()
                metric_tc = metric_batch[0].item()
                metric_wt = metric_batch[1].item()
                metric_et = metric_batch[2].item()
        
                # Reset metrics objects
                dice_metric_participant.reset()
                dice_metric_batch_participant.reset()
        
                writer.add_scalar(f"Validation/Personalized model/Client {client}/Mean_dice", metric_d, comm+1)
                writer.add_scalar(f"Validation/Personalized model/Client {client}/Tumor core dice", metric_tc, comm+1)
                writer.add_scalar(f"Validation/Personalized model/Client {client}/Whole tumor dice", metric_wt, comm+1)
                writer.add_scalar(f"Validation/Personalized model/Client {client}/Enhancing tumor dice", metric_et, comm+1)
                
                if metric_d > best_mean_dice[client]:
                    best_mean_dice[client] = metric_d
                    best_dice_et[client] = metric_et
                    best_dice_tc[client] = metric_tc
                    best_dice_wt[client] = metric_wt
                    best_comm[client] = comm + 1
                    # Save personalized models
                    torch.save(local_models[client].state_dict(), os.path.join(log_dir, f"best_local_model_client_{client}.pth"))                            
                    print("\nSaved new best metric personalized models")
                print(
                    f"\nClient {client}, Current epoch: {comm + 1}, current mean dice: {metric_d:.4f}"
                    f"\nmean dice tc: {metric_tc:.4f}"
                    f"\nmean dice wt: {metric_wt:.4f}"
                    f"\nmean dice et: {metric_et:.4f}"
                    f"\nbest mean dice: {best_mean_dice[client]:.4f}"
                    f"at global epoch: {best_comm[client]}"
                )
                
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
    
            writer.add_scalar("Validation/Personalized model/Mean_dice", metric_d, comm+1)
            writer.add_scalar("Validation/Personalized model/Tumor core dice", metric_tc, comm+1)
            writer.add_scalar("Validation/Personalized model/Whole tumor dice", metric_wt, comm+1)
            writer.add_scalar("Validation/Personalized model/Enhancing tumor dice", metric_et, comm+1)
    
            if (comm + 1) in config["save_comms"]:
                for client in samples_inst_map_train.keys():
                    shutil.copyfile(os.path.join(log_dir, f"best_local_model_client_{client}.pth"), os.path.join(log_dir, f"best_local_model_client_{client}_comm_{comm+1}.pth"))
            
    result_dict = {}
    for client in samples_inst_map_train.keys():
        result_dict[f"hparam/Mean_dice/Client {client}"] = best_mean_dice[client]
        result_dict[f"hparam/Dice_ET/Client {client}"] = best_dice_et[client]
        result_dict[f"hparam/Dice_TC/Client {client}"] = best_dice_tc[client]
        result_dict[f"hparam/Dice_WT/Client {client}"] = best_dice_wt[client]
        result_dict[f"hparam/Comm_round/Client {client}"] = best_comm[client]
        
    result_dict["hparam/Mean_dice"] = np.sum([best_mean_dice[client]*len(samples_inst_map_val[client]) for client in samples_inst_map_train.keys()])/len(val_dict)
    result_dict["hparam/Dice_ET"] = np.sum([best_dice_et[client]*len(samples_inst_map_val[client]) for client in samples_inst_map_train.keys()])/len(val_dict)
    result_dict["hparam/Dice_TC"] = np.sum([best_dice_tc[client]*len(samples_inst_map_val[client]) for client in samples_inst_map_train.keys()])/len(val_dict)
    result_dict["hparam/Dice_WT"] = np.sum([best_dice_wt[client]*len(samples_inst_map_val[client]) for client in samples_inst_map_train.keys()])/len(val_dict)
    
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
