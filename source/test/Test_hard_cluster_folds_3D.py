import os
import sys

currentdir = os.path.dirname(os.path.realpath(__file__))
parentdir = os.path.dirname(currentdir)
sys.path.append(parentdir)

import json
import torch
import monai
import matplotlib.pyplot as plt
import pandas as pd
from monai.config import print_config
from monai.data import DataLoader, decollate_batch, Dataset, LMDBDataset
from monai.losses import DiceLoss
from monai.metrics import DiceMetric, HausdorffDistanceMetric
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
   
    # Generating list of data files to load
    data_dir = config['data_dir']
    data_dict = generate_data_dict(data_dir)

    # Define the test transformation to be applied to data
    val_transform = generate_val_tranform(roi_size=config["roi_size"])
    post_trans = Compose(
        [EnsureType(), Activations(sigmoid=True), AsDiscrete(threshold_values=True)]
    )
 
    # Define the device to use (CPU or which GPU)
    device = torch.device(config["device"])
    
    batch_size = config["nb_gpus"]*config["batch_size_per_gpu"]
    
    # Define the metrics to be computed
    dice_metric = DiceMetric(include_background=True, reduction="mean", ignore_empty=False)
    dice_metric_batch = DiceMetric(include_background=True, reduction="mean_batch", ignore_empty=False)
    dice_metric_participant = DiceMetric(include_background=True, reduction="mean", ignore_empty=False)
    dice_metric_batch_participant = DiceMetric(include_background=True, reduction="mean_batch", ignore_empty=False)
    
    hausdorff_metric = HausdorffDistanceMetric(include_background=True, percentile=95, reduction="mean")
    hausdorff_metric_batch = HausdorffDistanceMetric(include_background=True, percentile=95, reduction="mean_batch")
    hausdorff_metric_participant = HausdorffDistanceMetric(include_background=True, percentile=95, reduction="mean")
    hausdorff_metric_batch_participant = HausdorffDistanceMetric(include_background=True, percentile=95, reduction="mean_batch")
    
    if config["test_folds"]:
        folds = config["folds"]
        
        if config["test_0"]:
            folds = [0] + folds
            
        for fold in folds:
            
            if fold == 0:
                current_dir = config["fold_0_dir"]
            else:
                current_dir = config["folds_dir"].replace('#', str(fold))
                
            # Loading test sets (we use the same function as for training, with a ratio of train=1.0)
            train_part_list, _, train_dict, _, \
                samples_inst_map_train, _ = gen_partitioning_fets(data_dict, 
                                                                    config["partition_file"].replace('#', str(fold)), 
                                                                    prop_full_dataset=config['prop_full_dataset'], 
                                                                    ratio_train=1.0)
            
            print(len(train_part_list[0]), len(samples_inst_map_train[list(samples_inst_map_train.keys())[0]]))
            
            # Copy config file in the experiment folder
            shutil.copy(args.config_path, current_dir)
            
            # Define the number of clients in the federation
            nb_clients = len(train_part_list)
                
            # Initialize dice metrics
            participant_metric_results = pd.DataFrame({"institution_id":pd.Series(dtype='int'),
                                                       "dataset_size":pd.Series(dtype='int'),
                                                       "participant_average_dices" : pd.Series(dtype='float'),
                                                       "participant_average_dices_tc" : pd.Series(dtype='float'),
                                                       "participant_average_dices_wt" : pd.Series(dtype='float'),
                                                       "participant_average_dices_et" : pd.Series(dtype='float'),
                                                       "participant_95_hausdorff_distances" : pd.Series(dtype='float'),
                                                       "participant_95_hausdorff_distances_tc" : pd.Series(dtype='float'),
                                                       "participant_95_hausdorff_distances_wt" : pd.Series(dtype='float'),
                                                       "participant_95_hausdorff_distances_et" : pd.Series(dtype='float')})
            
            aggregated_metric_results = pd.DataFrame({"weighted_average_dice" : pd.Series(dtype='float'),
                                           "weighted_average_dice_tc" : pd.Series(dtype='float'),
                                           "weighted_average_dice_wt" : pd.Series(dtype='float'),
                                           "weighted_average_dice_et" : pd.Series(dtype='float'),
                                           "uniform_average_dice" : pd.Series(dtype='float'),
                                           "uniform_average_dice_tc" : pd.Series(dtype='float'),
                                           "uniform_average_dice_wt" : pd.Series(dtype='float'),
                                           "uniform_average_dice_et" : pd.Series(dtype='float'),
                                           "worse_participant_dice_id" : pd.Series(dtype='int'),
                                           "worse_participant_dice" : pd.Series(dtype='float'),
                                           "worse_participant_dice_tc" : pd.Series(dtype='float'),
                                           "worse_participant_dice_wt" : pd.Series(dtype='float'),
                                           "worse_participant_dice_et" : pd.Series(dtype='float'),
                                           "weighted_95_hausdorff_distance" : pd.Series(dtype='float'),
                                           "weighted_95_hausdorff_distance_yc" : pd.Series(dtype='float'),
                                           "weighted_95_hausdorff_distance_wt" : pd.Series(dtype='float'),
                                           "weighted_95_hausdorff_distance_et" : pd.Series(dtype='float'),
                                           "uniform_95_hausdorff_distance" : pd.Series(dtype='float'),
                                           "uniform_95_hausdorff_distance_tc" : pd.Series(dtype='float'),
                                           "uniform_95_hausdorff_distance_wt" : pd.Series(dtype='float'),
                                           "uniform_95_hausdorff_distance_et" : pd.Series(dtype='float'),
                                           "worse_participant_hausdorff_id" : pd.Series(dtype='int'),
                                           "worse_95_hausdorff_distance" : pd.Series(dtype='float'),
                                           "worse_95_hausdorff_distance_tc" : pd.Series(dtype='float'),
                                           "worse_95_hausdorff_distance_wt" : pd.Series(dtype='float'),
                                           "worse_95_hausdorff_distance_et" : pd.Series(dtype='float')})
            
            sample_results = pd.DataFrame({"sample": pd.Series(dtype='str'),
                                           "institution": pd.Series(dtype='str'),
                                           "average_dice" : pd.Series(dtype='float'),
                                           "dice_tc" : pd.Series(dtype='float'),
                                           "dice_wt" : pd.Series(dtype='float'),
                                           "dice_et" : pd.Series(dtype='float'),
                                           "average_95_hausdorff_distance" : pd.Series(dtype='float'),
                                           "95_hausdorff_distance_tc" : pd.Series(dtype='float'),
                                           "95_hausdorff_distance_wt" : pd.Series(dtype='float'),
                                           "95_hausdorff_distance_et" : pd.Series(dtype='float')})
            
            # Load the model trained on remaining samples
            monai.networks.blocks.dynunet_block.get_conv_layer = modified_get_conv_layer
            local_model = DynUNet(
                spatial_dims = 3,
                in_channels = 4,
                out_channels = 3,
                kernel_size = config["kernel_sizes"],
                filters = config["filters"],
                strides = config["strides"],
                upsample_kernel_size = config["strides"][1:],
                norm_name=("INSTANCE", {"affine": False}),
                act_name=("LeakyReLu", {"negative_slope":0.01}),
                trans_bias=True,
            ).to(device)
            
            # Find clusters for the current fold
            clusters = config["clusters"][str(fold)]
            
            # Computation of test metrics
            for list_clients in clusters:
            
                #local_model.load_state_dict(torch.load(os.path.join(current_dir, config["model_file"].replace("#", str(tuple(list_clients))))))
                local_model.load_state_dict(torch.load(os.path.join(current_dir, config["model_file"].replace("#", str(list_clients)))))
                
                for name, param in local_model.named_parameters():
                    print(name, param.shape)
                    
                if config["multi_gpu"]:
                    model_parallel = torch.nn.DataParallel(local_model)
                else:
                    model_parallel = local_model 
                    
                local_model.eval()
                
                # Computation of test metrics
                for client in list_clients:
                    
                    with torch.no_grad(): 
                        
                        for (idx, val_data) in enumerate(tqdm(samples_inst_map_train[client])):
                            
                            transformed_val_data = val_transform(val_data)
                            
                            val_inputs, val_labels = (
                                transformed_val_data["image"].unsqueeze(0).to(device),
                                transformed_val_data["label"].unsqueeze(0).to(device),
                            )
                            
                            # Sliding window inference and thresholding
                            val_outputs = sliding_window_inference(val_inputs, config["roi_size"], batch_size, model_parallel, overlap=config["sliding_window_overlap"], mode="gaussian")
                            val_outputs = [post_trans(i) for i in decollate_batch(val_outputs)]
            
                            # Compute metric for current iteration
                            dice_metric(y_pred=val_outputs, y=val_labels)
                            dice_metric_batch(y_pred=val_outputs, y=val_labels)
                            dice_metric_participant(y_pred=val_outputs, y=val_labels)
                            dice_metric_batch_participant(y_pred=val_outputs, y=val_labels)
                            
                            hausdorff_metric(y_pred=val_outputs, y=val_labels)
                            hausdorff_metric_batch(y_pred=val_outputs, y=val_labels)
                            hausdorff_metric_participant(y_pred=val_outputs, y=val_labels)
                            hausdorff_metric_batch_participant(y_pred=val_outputs, y=val_labels)
                            
                            print(dice_metric.get_buffer().shape)
                            
                            sample_results.loc[len(sample_results.index)] = \
                                [
                                    os.path.basename(val_data["label"]),
                                    client,
                                    dice_metric.get_buffer()[-1].mean().item(),
                                    dice_metric.get_buffer()[-1][0].item(),
                                    dice_metric.get_buffer()[-1][1].item(),
                                    dice_metric.get_buffer()[-1][2].item(),
                                    hausdorff_metric.get_buffer()[-1].mean().item(),
                                    hausdorff_metric.get_buffer()[-1][0].item(),
                                    hausdorff_metric.get_buffer()[-1][1].item(),
                                    hausdorff_metric.get_buffer()[-1][2].item()
                                ]
                                     
                    participant_dice = dice_metric_participant.aggregate().item()
                    participant_dice_batch = dice_metric_batch_participant.aggregate()
                    participant_dice_tc = participant_dice_batch[0].item()
                    participant_dice_wt = participant_dice_batch[1].item()
                    participant_dice_et = participant_dice_batch[2].item()
                    
                    participant_hausdorff = hausdorff_metric_participant.aggregate().item()
                    participant_hausdorff_batch = hausdorff_metric_batch_participant.aggregate()
                    participant_hausdorff_tc = participant_hausdorff_batch[0].item()
                    participant_hausdorff_wt = participant_hausdorff_batch[1].item()
                    participant_hausdorff_et = participant_hausdorff_batch[2].item()
                    
                    dice_metric_participant.reset()
                    dice_metric_batch_participant.reset()
                    hausdorff_metric_participant.reset()
                    hausdorff_metric_batch_participant.reset()
                    
                    participant_metric_results.loc[len(participant_metric_results.index)] = \
                        [
                            client,
                            len(samples_inst_map_train[client]),
                            participant_dice,
                            participant_dice_tc,
                            participant_dice_wt,
                            participant_dice_et,
                            participant_hausdorff,
                            participant_hausdorff_tc,
                            participant_hausdorff_wt,
                            participant_hausdorff_et
                        ]
            
            print(participant_metric_results.dtypes)
                
            # Aggregate the final mean metrics results
            weighted_dice = dice_metric.aggregate().item()
            weighted_hausdorff = hausdorff_metric.aggregate().item()
            
            # Aggregate the final mean metrics for each label
            weighted_dice_batch = dice_metric_batch.aggregate()
            weighted_dice_batch_tc = weighted_dice_batch[0].item()
            weighted_dice_batch_wt = weighted_dice_batch[1].item()
            weighted_dice_batch_et = weighted_dice_batch[2].item()
            
            weighted_hausdorff_batch = hausdorff_metric_batch.aggregate()
            weighted_hausdorff_batch_tc = weighted_hausdorff_batch[0].item()
            weighted_hausdorff_batch_wt = weighted_hausdorff_batch[1].item()
            weighted_hausdorff_batch_et = weighted_hausdorff_batch[2].item()
        
            # Reset metrics objects
            dice_metric.reset()
            dice_metric_batch.reset()
            hausdorff_metric.reset()
            hausdorff_metric_batch.reset()     
                 
            worse_dice_index = participant_metric_results["participant_average_dices"].idxmin()
            print(worse_dice_index)
            worse_hausdorff_index = participant_metric_results["participant_95_hausdorff_distances"].idxmax()
            print(worse_hausdorff_index)
            
            aggregated_metric_results.loc[len(aggregated_metric_results)] = \
                [
                    weighted_dice,
                    weighted_dice_batch_tc,
                    weighted_dice_batch_wt,
                    weighted_dice_batch_et,
                    participant_metric_results["participant_average_dices"].mean(),
                    participant_metric_results["participant_average_dices_tc"].mean(),
                    participant_metric_results["participant_average_dices_wt"].mean(),
                    participant_metric_results["participant_average_dices_et"].mean(),
                    participant_metric_results.iloc[worse_dice_index]["institution_id"],
                    participant_metric_results.iloc[worse_dice_index]["participant_average_dices"],
                    participant_metric_results.iloc[worse_dice_index]["participant_average_dices_tc"],
                    participant_metric_results.iloc[worse_dice_index]["participant_average_dices_wt"],
                    participant_metric_results.iloc[worse_dice_index]["participant_average_dices_et"],
                    weighted_hausdorff,
                    weighted_hausdorff_batch_tc,
                    weighted_hausdorff_batch_wt,
                    weighted_hausdorff_batch_et,
                    participant_metric_results["participant_95_hausdorff_distances"].mean(),
                    participant_metric_results["participant_95_hausdorff_distances_tc"].mean(),
                    participant_metric_results["participant_95_hausdorff_distances_wt"].mean(),
                    participant_metric_results["participant_95_hausdorff_distances_et"].mean(),
                    participant_metric_results.iloc[worse_hausdorff_index]["institution_id"],
                    participant_metric_results.iloc[worse_hausdorff_index]["participant_95_hausdorff_distances"],
                    participant_metric_results.iloc[worse_hausdorff_index]["participant_95_hausdorff_distances_tc"],
                    participant_metric_results.iloc[worse_hausdorff_index]["participant_95_hausdorff_distances_wt"],
                    participant_metric_results.iloc[worse_hausdorff_index]["participant_95_hausdorff_distances_et"]
                ]
            
            participant_metric_results.to_csv(os.path.join(current_dir, f"participant_metric_results.csv"))
            aggregated_metric_results.to_csv(os.path.join(current_dir, f"aggregated_metric_results.csv"))
            sample_results.to_csv(os.path.join(current_dir, f"sample_metric_results.csv"))
        
    # Compute aggregated metrics on all folds
    if config["final_metrics_all_folds"]:
        folds = [0, 1, 2, 3, 4]
    
        full_sample_results = None
        
        for fold in folds:
            
            if fold == 0:
                current_dir = config["fold_0_dir"]
            else:
                current_dir = config["folds_dir"].replace('#', str(fold))
                
            if full_sample_results is None:
                full_sample_results = pd.read_csv(os.path.join(current_dir, "sample_metric_results.csv"))
            else:
                full_sample_results = full_sample_results.append(pd.read_csv(os.path.join(current_dir, "sample_metric_results.csv")), ignore_index = True)
                
        print(full_sample_results)
        print("Size of final sample results (should be 1251): ", len(full_sample_results))
        print("Nan 95hd et: ", full_sample_results[full_sample_results["95_hausdorff_distance_et"].isna()])
        print("Number of duplicate sample names (should be 0): ", full_sample_results["sample"].duplicated().sum())
        aggregated_columns = ["average_dice", "dice_tc", "dice_wt", "dice_et", 
                              "average_95_hausdorff_distance", "95_hausdorff_distance_tc", "95_hausdorff_distance_wt", "95_hausdorff_distance_et"]
        
        save_dir = os.path.dirname(config["folds_dir"])
        
        # Create aggregated results (avg and std over all samples)
        final_metrics = pd.DataFrame()
        
        for metric in aggregated_columns:
            final_metrics[metric+"_full_avg"] = [full_sample_results[metric].mean()]
            final_metrics[metric+"_full_std"] = [full_sample_results[metric].std()]
                
        # Create results aggregated per institution (avg and std)
        final_participant_metrics = pd.DataFrame()
        
        institutions = full_sample_results["institution"].unique()
        
        for idx, inst in enumerate(institutions):
            final_participant_metrics.loc[idx, "institution"] = inst
            
            inst_samples_results = full_sample_results[full_sample_results["institution"]==inst]
            final_participant_metrics.loc[idx, "dataset_size"] = inst_samples_results["sample"].count()
            
            for metric in aggregated_columns:
                final_participant_metrics.loc[idx, metric+"_full_avg"] = inst_samples_results[metric].mean()
                final_participant_metrics.loc[idx, metric+"_full_std"] = inst_samples_results[metric].std()
            
        full_sample_results.to_csv(os.path.join(save_dir, "all_folds_sample_metrics.csv"))
        final_participant_metrics.to_csv(os.path.join(save_dir, "all_folds_participants_metrics.csv"))
        final_metrics.to_csv(os.path.join(save_dir, "all_folds_aggregated_metrics.csv"))
        
        
            
            
    

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