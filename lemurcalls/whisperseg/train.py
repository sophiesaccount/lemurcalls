import argparse
import json
import os
import sys
from datetime import datetime
from glob import glob
import matplotlib.pyplot as plt

import numpy as np
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import get_linear_schedule_with_warmup

from .convert_hf_to_ct2 import convert_hf_to_ct2
from .datautils_ben import (VocalSegDataset, get_audio_and_label_paths, get_audio_and_label_paths_from_folders,
                       get_cluster_codebook, load_data,
                       slice_audios_and_labels, train_val_split,
                       get_codebook_for_classes)
from .model import WhisperSegmenterForEval, load_model, save_model
from ..util.common import EarlyStopHandler, is_scheduled_job
from ..util.confusion_framewise import confusion_matrix_framewise
from ..utils import *


def collate_fn(batch):
    """Stack batch items into tensors for input_features, decoder_input_ids, labels.

    Args:
        batch: List of sample dicts from VocalSegDataset.

    Returns:
        dict: Batched tensors for input_features, decoder_input_ids, labels.
    """
    # input_features = [item["input_features"].clone().detach().float() for item in batch]
    #decoder_input_ids = [item["decoder_input_ids"].clone().detach().long() for item in batch]
    #labels = [item["labels"].clone().detach().long() for item in batch]

    input_features = [torch.tensor(item["input_features"], dtype=torch.float32) for item in batch]
    decoder_input_ids = [torch.tensor(item["decoder_input_ids"], dtype=torch.int64) for item in batch]
    labels = [torch.tensor(item["labels"], dtype=torch.int64) for item in batch]

    # Stack tensors along the first dimension (batch dimension)
    input_features = torch.stack(input_features)
    decoder_input_ids = torch.stack(decoder_input_ids)
    labels = torch.stack(labels)

    return {
        "input_features": input_features,
        "decoder_input_ids": decoder_input_ids,
        "labels": labels,
    }


def train_iteration(batch):
    """Run one training step (forward, loss, backward, step). Uses global model, optimizer, scaler, device."""
    for key in batch:
        batch[key] = batch[key].to(device)
    
    optimizer.zero_grad()
    with torch.amp.autocast(device_type="cuda", dtype=torch.float16):   
        model_out = model( **batch )
        loss = model_out.loss.mean()
    scaler.scale(loss).backward()
    scaler.step(optimizer)
    # optimizer.step()
    scaler.update()
    return loss.item()

def evaluate(audio_list, label_list, segmenter, batch_size, max_length, num_trials, consolidation_method="clustering", num_beams=4, target_cluster=None, confusion_matrix: str = None, save_cm_data: str = None):
    """Run segmenter on each audio and aggregate segment-wise and frame-wise precision/recall/F1.

    Args:
        audio_list: List of audio arrays.
        label_list: List of label dicts (with sr, min_frequency, spec_time_step, etc.).
        segmenter: Segmenter instance with .segment(), .segment_score(), .frame_score().
        batch_size: Batch size for segmenter.segment().
        max_length: Max generation length.
        num_trials: Number of trials for multi-trial segmentation.
        consolidation_method: 'clustering' or voting method.
        num_beams: Beam size.
        target_cluster: If set, score only this cluster; else all.
        confusion_matrix: Optional name to compute/save confusion matrix.
        save_cm_data: Optional name to save raw prediction/label for confusion analysis.

    Returns:
        dict: 'segment_wise' and 'frame_wise' lists [TP, P_pred, P_label, precision, recall, f1].
    """
    total_n_true_positive_segment_wise, total_n_positive_in_prediction_segment_wise, total_n_positive_in_label_segment_wise = 0,0,0
    total_n_true_positive_frame_wise, total_n_positive_in_prediction_frame_wise, total_n_positive_in_label_frame_wise = 0,0,0

    for audio, label in tqdm(zip(audio_list, label_list), total = len(audio_list), desc = "evaluate()", disable=is_scheduled_job()):
        prediction = segmenter.segment(
            audio, sr = 48000,
            min_frequency = 0,
            spec_time_step = 0.0025,
            min_segment_length = 0.0195,
            eps = 0.02,  ## for DBSCAN clustering
            time_per_frame_for_voting = label.get("time_per_frame_for_voting", 0.001), ## for bin-wise voting, by default it is not used
            consolidation_method = consolidation_method,
            max_length = max_length, 
            batch_size = batch_size, 
            num_trials = num_trials,
            num_beams = num_beams
        )
        # Workaround to pass job-id via confusion_matrix and save_cm_data
        if confusion_matrix != None:
            confusion_matrix_framewise(prediction, label, None, 0.001, name=confusion_matrix)
        if save_cm_data != None:
            with open(f'/usr/users/bhenne/projects/whisperseg/results/{save_cm_data}.cmraw', "w") as fp:
                json.dump({'prediction': prediction, 'label': label}, fp, indent=2)
        TP, P_pred, P_label = segmenter.segment_score( prediction, label,  target_cluster = target_cluster, tolerance = 0.02 )[:3]
        total_n_true_positive_segment_wise += TP
        total_n_positive_in_prediction_segment_wise += P_pred
        total_n_positive_in_label_segment_wise += P_label
        
        
        TP, P_pred, P_label = segmenter.frame_score( prediction, label,  target_cluster = target_cluster, 
                                                     time_per_frame_for_scoring = 0.001 )[:3]
        
        total_n_true_positive_frame_wise += TP
        total_n_positive_in_prediction_frame_wise += P_pred
        total_n_positive_in_label_frame_wise += P_label
        
    res = {}
    
    precision = total_n_true_positive_segment_wise / max(total_n_positive_in_prediction_segment_wise, 1e-12)
    recall = total_n_true_positive_segment_wise / max( total_n_positive_in_label_segment_wise, 1e-12 )
    f1 = 2/(1/max(precision, 1e-12) + 1/max(recall, 1e-12)  )
    
    res["segment_wise"] = [ total_n_true_positive_segment_wise, total_n_positive_in_prediction_segment_wise, total_n_positive_in_label_segment_wise, precision, recall, f1 ]
    
    
    precision = total_n_true_positive_frame_wise / max(total_n_positive_in_prediction_frame_wise, 1e-12)
    recall = total_n_true_positive_frame_wise / max( total_n_positive_in_label_frame_wise, 1e-12 )
    f1 = 2/(1/max(precision, 1e-12) + 1/max(recall, 1e-12)  )
    
    res["frame_wise"] = [ total_n_true_positive_frame_wise, total_n_positive_in_prediction_frame_wise, total_n_positive_in_label_frame_wise, precision, recall, f1 ]
    
    return res

if __name__ == "__main__":
    
    parser = argparse.ArgumentParser()

    parser.add_argument("--initial_model_path" )
    parser.add_argument("--train_dataset_folder", default=None )
    parser.add_argument("--model_folder" )
    parser.add_argument("--audio_folder" )
    parser.add_argument("--label_folder" )
    parser.add_argument("--n_device", type = int, default = 1 )
    parser.add_argument("--gpu_list", type = int, nargs = "+", default = None )
    parser.add_argument("--update_every", type = int, default = 100 )
    parser.add_argument("--validate_every", type = int, default = None )
    parser.add_argument("--validate_per_epoch", type = int, default = 0 )
    parser.add_argument("--save_every", type = int, default = None )
    parser.add_argument("--save_per_epoch", type = int, default = 0 )
    parser.add_argument("--max_num_epochs", type = int, default = 3 )
    parser.add_argument("--max_num_iterations", type = int, default = None )
    parser.add_argument("--val_ratio", type = float, default = 0.2 )
    parser.add_argument("--patience", type = int, default = 10, help="If the validation score does not improve for [patience] epochs, stop training.")
    
    parser.add_argument("--max_length", type = int, default = 100 )
    parser.add_argument("--total_spec_columns", type = int, default = 1000 )
    parser.add_argument("--batch_size", type = int, default = 4 )
    parser.add_argument("--learning_rate", type = float, default = 3e-6 )
    parser.add_argument("--lr_schedule", default = "linear" )
    parser.add_argument("--max_to_keep", type = int, default = -1 )
    parser.add_argument("--seed", type = int, default = 66100 )
    parser.add_argument("--weight_decay", type = float, default = 0.01 )
    parser.add_argument("--warmup_steps", type = int, default = 100 )
    parser.add_argument("--freeze_encoder", type = int, default = 0 )
    parser.add_argument("--dropout", type = float, default = 0.0 )
    parser.add_argument("--num_workers", type = int, default = 4 )
    parser.add_argument("--clear_cluster_codebook", type = int, help="set the pretrained model's cluster_codebook to empty dict. This is used when we train the segmenter on a complete new dataset. Set this to 0 if you just want to slighlt finetune the model with some additional data with the same cluster naming rule.", default = 0 )
    parser.add_argument("--num_classes", type = int, default = None,
                        help="Number of output classes. 1 = single-class (moan), 3 = multi-class (m/h/w). "
                             "If not set, the codebook is built dynamically from training labels.")
    
    args = parser.parse_args()

    if args.seed is not None:
        np.random.seed(args.seed)  
        
    if args.val_ratio == 0.0:
        args.validate_every = None
        args.validate_per_epoch= None

    create_if_not_exists(args.model_folder)

    if args.gpu_list is None:
        args.gpu_list = np.arange(args.n_device).tolist()
        
    device = torch.device(  "cuda:%d"%( args.gpu_list[0] ) if torch.cuda.is_available() else "cpu" )

    model, tokenizer = load_model( args.initial_model_path, args.total_spec_columns, args.dropout)

    model = model.to(device)
    
    if args.freeze_encoder:
        for para in model.model.encoder.parameters():
            para.requires_grad = False
    else:
        for para in model.model.encoder.parameters():
            para.requires_grad = True
            
    no_decay = ['bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [
        {'params': [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)], 'weight_decay': args.weight_decay},
        {'params': [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
    ]
    optimizer = AdamW(optimizer_grouped_parameters, lr = args.learning_rate )
    
    model = nn.DataParallel( model, args.gpu_list )

    segmenter = WhisperSegmenterForEval( model = model, tokenizer = tokenizer )

    if args.clear_cluster_codebook:
        segmenter.update_cluster_codebook( {} )

    scaler = torch.cuda.amp.GradScaler()

    #audio_path_list_train, label_path_list_train = get_audio_and_label_paths( args.train_dataset_folder )  

    if args.audio_folder and args.label_folder:
        audio_path_list_train, label_path_list_train = get_audio_and_label_paths_from_folders(
            args.audio_folder, args.label_folder)
    else:
        audio_path_list_train, label_path_list_train = get_audio_and_label_paths(args.train_dataset_folder)

    if args.num_classes is not None:
        cluster_codebook = get_codebook_for_classes(args.num_classes)
        print(f"Using fixed codebook for {args.num_classes} class(es): {cluster_codebook}")
    else:
        cluster_codebook = get_cluster_codebook( label_path_list_train, segmenter.cluster_codebook )
        print(f"Built dynamic codebook from labels: {cluster_codebook}")
    segmenter.update_cluster_codebook( cluster_codebook )

    audio_list_train, label_list_train = load_data(audio_path_list_train, label_path_list_train, cluster_codebook = cluster_codebook, n_threads = 20 )

    if args.val_ratio > 0:
        (audio_list_train, label_list_train), ( audio_list_val, label_list_val ) = train_val_split( audio_list_train, label_list_train, args.val_ratio )

    audio_list_train, label_list_train = slice_audios_and_labels( audio_list_train, label_list_train, args.total_spec_columns )

    # Check if we have any data after slicing
    if len(audio_list_train) == 0:
        print("Error: No valid audio samples after slicing!")
        print("This could be due to:")
        print("  - Audio files that are too short after slicing")
        print("  - No valid segments in the labels")
        print("  - All segments being filtered out during processing")
        sys.exit(1)

    print(f"Created {len(audio_list_train)} training samples after slicing")

    training_dataset = VocalSegDataset( audio_list_train, label_list_train, tokenizer, args.max_length, 
                                         args.total_spec_columns, model.module.config.species_codebook  )

    # Check dataset size before creating DataLoader
    if len(training_dataset) == 0:
        print("Error: Training dataset has 0 samples!")
        sys.exit(1)
    
    if len(training_dataset) < args.batch_size:
        print(f"Warning: Dataset size ({len(training_dataset)}) is smaller than batch size ({args.batch_size})")
        print("Consider reducing batch size or adding more data")

    training_dataloader = DataLoader( training_dataset, batch_size = args.batch_size , shuffle = True , 
                                             worker_init_fn = lambda x:[np.random.seed( epoch  + x ),  
                                                                    torch.manual_seed( epoch + x) ], 
                                             num_workers = args.num_workers , drop_last= True,
                                             pin_memory = False,
                                             collate_fn = collate_fn
                                           )

    if len(training_dataloader) == 0:
        print("Error: Too few examples (less than a batch) for training! Exit!")
        sys.exit(1)
    

    if args.max_num_iterations is not None and args.max_num_iterations > 0:
        args.max_num_epochs = int(np.ceil( args.max_num_iterations / len( training_dataloader )  ))
    else:
        assert args.max_num_epochs is not None and args.max_num_epochs > 0
        args.max_num_iterations = len( training_dataloader ) * args.max_num_epochs
                
    if args.lr_schedule == "linear":
        scheduler = get_linear_schedule_with_warmup(
            optimizer, num_warmup_steps= args.warmup_steps, 
            num_training_steps = args.max_num_iterations
        )
    else:
        scheduler = None
        
    model.train() 
    training_loss_value_list = []
    val_score_history = []
    esh = EarlyStopHandler(patience = args.patience)
    early_stop = False
    current_step = 0


    for epoch in range(args.max_num_epochs + 1):  # This +1 is to ensure current_step can reach args.max_num_iterations
        for count, batch in enumerate( tqdm( training_dataloader, desc=f'epoch-{epoch:03}', disable=is_scheduled_job()) ):
            training_loss_value_list.append( train_iteration(batch) )
            
            if scheduler is not None:
                scheduler.step()
                
            current_step += 1

            if args.update_every > 0 and current_step % args.update_every == 0:
                training_loss_value_list = [] 

            if ( args.validate_every is not None and current_step % args.validate_every == 0 ) or \
                ( args.validate_per_epoch and count == len(training_dataloader) - 1 ):
                model.eval()
                ## in the validation set, set the num_trails to 1
                eval_res = evaluate( audio_list_val, label_list_val, segmenter, args.batch_size, args.max_length, num_trials =1, consolidation_method = None, num_beams=1, target_cluster = None )
                val_score_history.append( ( current_step, ( eval_res["segment_wise"][-1] + eval_res["frame_wise"][-1] ) * 0.5 ) )
                early_stop = esh.check(val_score_history[-1][1]) if len(val_score_history) > 0 else False
                model.train()
            
            if ( args.save_every is not None and current_step % args.save_every == 0 ) or \
               ( args.save_per_epoch and count == len(training_dataloader) - 1 ):
                model.eval()
                save_model( model, tokenizer, current_step, args.model_folder, args.max_to_keep )
                model.train()

            if current_step >= args.max_num_iterations or early_stop :
                if not os.path.exists( args.model_folder+"/checkpoint-%d"%(current_step) ):
                    model.eval()
                    save_model( model, tokenizer, current_step, args.model_folder, args.max_to_keep )
                break

        if current_step >= args.max_num_iterations or early_stop :
            break   

    best_checkpoint_batch_number = None
    if len(val_score_history) > 0:
        best_checkpoint_batch_number = sorted( val_score_history, key = lambda x:-x[1] )[0][0]
    else:
        ckpt_list = glob( args.model_folder + "/*" )
        if len( ckpt_list ) >0:
            ckpt_list.sort( key = os.path.getmtime )
            ckpt_name = ckpt_list[-1]
            best_checkpoint_batch_number = int(ckpt_name.split("-")[-1])
        

    if best_checkpoint_batch_number is not None:
        print("The best checkpoint on validation set is: %s," % ( args.model_folder+"/checkpoint-%d"%(best_checkpoint_batch_number) ) )
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        hf_model_folder = f"{args.model_folder}/final_checkpoint_{timestamp}"
        ct2_model_folder = hf_model_folder + "_ct2"

        os.system( "cp -r %s %s"%( args.model_folder+"/checkpoint-%d"%(best_checkpoint_batch_number), hf_model_folder ) )
        ### remove other checkpoints
        os.system( "rm -r %s"%( args.model_folder+"/checkpoint-*" ) )

        convert_hf_to_ct2(model=hf_model_folder, output_dir=ct2_model_folder, quantization="float16")

        params_path = os.path.join(hf_model_folder, "training_args.json")
        with open(params_path, "w") as f:
            json.dump(vars(args), f, indent=4)
        


        # === Plot Loss Curves ===
        plt.figure(figsize=(8, 5))
        plt.plot(training_loss_value_list, label="Training Loss")
        plt.plot(val_score_history, label="Validation Loss")

        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.title("Training and Validation Loss")
        plt.legend()
        plt.grid(True)
        plt.tight_layout()

        # Save plot
        loss_plot_path = os.path.join(ct2_model_folder, "loss_curve.png")
        plt.savefig(loss_plot_path)
        print("Saved loss curve to:", loss_plot_path)
        plt.close()

        print("All Done!")
