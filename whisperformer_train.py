### used to check things like data loader


import argparse
import json
import os
import sys
from datetime import datetime
from glob import glob

import numpy as np
import torch
import torch.nn as nn
import wandb
from torch.optim import AdamW
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import get_linear_schedule_with_warmup

from convert_hf_to_ct2 import convert_hf_to_ct2
from datautils import (VocalSegDataset, get_audio_and_label_paths, get_audio_and_label_paths_from_folders,
                       get_cluster_codebook, load_data,
                       slice_audios_and_labels, train_val_split)
from model import WhisperSegmenterForEval, load_model, save_model
from util.common import EarlyStopHandler, is_scheduled_job
from util.confusion_framewise import confusion_matrix_framewise
from utils import *
from torch.nn.utils.rnn import pad_sequence
from whisperformer_dataset import WhisperFormerDataset
from whisperformer_model import WhisperFormer
from losses import sigmoid_focal_loss, ctr_diou_loss_1d


def losses(out_cls, out_offsets, gt_cls, gt_offsets, loss_normalizer=200, loss_normalizer_momentum=0.9, train_loss_weight=0):

    #get positive mask: points with true positives
    pos_mask = gt_cls > 0

    # update the loss normalizer #ToDo: problem that we now have offsets for all classes?
    num_pos = pos_mask.sum().item()
    # EMA
    loss_normalizer = loss_normalizer_momentum * loss_normalizer + (
        1 - loss_normalizer_momentum
    ) * max(num_pos, 1)

    # focal loss
    cls_loss = sigmoid_focal_loss(out_cls, gt_cls, reduction='sum')
    cls_loss /= loss_normalizer
    
    # if there are no positive samples, set regression loss to zero
    if num_pos == 0:
        reg_loss = 0 * out_offsets.sum()
    else:
        # giou loss defined on positive samples
        reg_loss = ctr_diou_loss_1d(out_offsets[pos_mask], gt_offsets[pos_mask], reduction='sum')
        reg_loss /= loss_normalizer

    if train_loss_weight > 0:
        loss_weight = train_loss_weight
    else:
        loss_weight = cls_loss.detach() / max(reg_loss.item(), 0.01)

    # return a dict of losses
    final_loss = cls_loss + reg_loss * loss_weight
    return {'cls_loss'   : cls_loss,
            'reg_loss'   : reg_loss,
            'final_loss' : final_loss}



def load_actionformer_model(initial_model_path, num_classes):
    """Load ActionFormer model with Whisper encoder"""
    from transformers import WhisperModel
    
    # Load Whisper encoder #????this should be part of the model already!
    whisper_model = WhisperModel.from_pretrained("openai/whisper-tiny")
    encoder = whisper_model.encoder
    
    # Create ActionFormer model with the correct number of classes
    model = WhisperFormer(encoder, num_classes=num_classes)
    
    # Load pretrained weights if available
    if initial_model_path and os.path.exists(initial_model_path):
        print(f"Loading pretrained weights from {initial_model_path}")
        model.load_state_dict(torch.load(initial_model_path, map_location='cpu'))
    
    return model


def actionformer_train_iteration(model, batch, optimizer, scheduler, scaler, device):
    """Training iteration for ActionFormer model"""
    for key in batch:
        batch[key] = batch[key].to(device)
    
    optimizer.zero_grad()
    
    total_loss = 0
    with torch.amp.autocast(device_type="cuda", dtype=torch.float16):
        # Forward pass
        class_preds, regr_preds = model(batch["input_features"])
        #print(class_preds, regr_preds)

        
        # Calculate losses
        cls_loss, reg_loss, total_loss = losses(class_preds, regr_preds, batch['clusters'], batch['segments'], loss_normalizer=200, loss_normalizer_momentum=0.9, train_loss_weight=0)


        #cls_loss = sigmoid_focal_loss(class_preds, batch["clusters"])
        #print(cls_loss)
        #regr_loss = ctr_diou_loss_1d(regr_preds, batch['segments']) #VORSICHT: nur für die valid ones!
        #print(regr_loss)
        
        #total_loss += (cls_loss + 0.1 * regr_loss).item()
    
    #scaler.scale(cls_loss.mean()).backward()
    #scaler.step(optimizer)
    #scaler.update()
    
    #return total_loss.item()
    return total_loss


###FEHLER!!!!#####
def actionformer_validation_loss(model, val_dataloader, device):
    """Compute validation loss for ActionFormer model"""
    model.eval()
    total_loss_accum = 0.0
    batch_count = 0
    
    with torch.no_grad():
        for batch in val_dataloader:
            for key in batch:
                batch[key] = batch[key].to(device)
            
        with torch.amp.autocast(device_type="cuda", dtype=torch.float16):
            class_preds, regr_preds = model(batch["input_features"])
            
            # Calculate losses (same as training) #TODO
            cls_loss, reg_loss, total_loss = losses(class_preds, regr_preds, batch['clusters'], batch['segments'], loss_normalizer=200, loss_normalizer_momentum=0.9, train_loss_weight=0)
            
            total_loss_accum += total_loss
            batch_count += 1

    model.train()
    return total_loss_accum / batch_count


def collate_fn(batch):
    # batch is a list of samples (dicts)
    
    input_features = [item["input_features"].clone().detach().float() for item in batch]
    segments = torch.stack([item["segments"] for item in batch])
    clusters = torch.stack([item["clusters"] for item in batch])


    input_features = torch.stack(input_features) 
    return {
        "input_features": input_features,       # [B, F, T] for B=batch_size, F=feature_dim, T=total_spec_columns
        "segments": segments,                   # [B, T , 2]
        "clusters": clusters,                   # [B, T, C]
    }



def train_iteration(batch):
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

def evaluate( audio_list, label_list, segmenter, batch_size, max_length, num_trials, consolidation_method = "clustering", num_beams=4, target_cluster = None, confusion_matrix: str = None, save_cm_data: str = None):

    total_n_true_positive_segment_wise, total_n_positive_in_prediction_segment_wise, total_n_positive_in_label_segment_wise = 0,0,0
    total_n_true_positive_frame_wise, total_n_positive_in_prediction_frame_wise, total_n_positive_in_label_frame_wise = 0,0,0

    for audio, label in tqdm(zip(audio_list, label_list), total = len(audio_list), desc = "evaluate()", disable=is_scheduled_job()):
        prediction = segmenter.segment(
            audio, sr = label["sr"],
            min_frequency = label["min_frequency"],
            spec_time_step = label["spec_time_step"],
            min_segment_length = label["min_segment_length"],
            eps = label["eps"],  ## for DBSCAN clustering
            time_per_frame_for_voting = label.get("time_per_frame_for_voting", 0.001), ## for bin-wise voting, by default it is not used
            consolidation_method = consolidation_method,
            max_length = max_length, 
            batch_size = batch_size, 
            num_trials = num_trials,
            num_beams = num_beams
        )
        # dirty workaround to pass the job-id in `confusion_matrix` and `save_cm_data
        if confusion_matrix != None:
            confusion_matrix_framewise(prediction, label, None, label["time_per_frame_for_scoring"], name=confusion_matrix)
        if save_cm_data != None:
            with open(f'/usr/users/bhenne/projects/whisperseg/results/{save_cm_data}.cmraw', "w") as fp:
                json.dump({'prediction': prediction, 'label': label}, fp, indent=2)
        TP, P_pred, P_label = segmenter.segment_score( prediction, label,  target_cluster = target_cluster, tolerance = label["tolerance"] )[:3]
        total_n_true_positive_segment_wise += TP
        total_n_positive_in_prediction_segment_wise += P_pred
        total_n_positive_in_label_segment_wise += P_label
        
        
        TP, P_pred, P_label = segmenter.frame_score( prediction, label,  target_cluster = target_cluster, 
                                                     time_per_frame_for_scoring = label["time_per_frame_for_scoring"] )[:3]
        
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
    parser.add_argument("--project", default = "wseg-lemur" )
    parser.add_argument("--run_name", default = None )
    parser.add_argument("--run_notes", default = None )
    parser.add_argument("--run_tags", default = None, nargs='+')
    parser.add_argument("--wandb_dir", default=None)
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
    parser.add_argument("--total_spec_columns", type = int, default = 3000 )
    parser.add_argument("--batch_size", type = int, default = 4 )
    parser.add_argument("--learning_rate", type = float, default = 3e-6 )
    parser.add_argument("--lr_schedule", default = "linear" )
    parser.add_argument("--max_to_keep", type = int, default = -1 )
    parser.add_argument("--seed", type = int, default = 66100 )
    parser.add_argument("--weight_decay", type = float, default = 0.01 )
    parser.add_argument("--warmup_steps", type = int, default = 100 )
    parser.add_argument("--freeze_encoder", type = bool, default = True)
    parser.add_argument("--dropout", type = float, default = 0.0 )
    parser.add_argument("--num_workers", type = int, default = 4 )
    parser.add_argument("--clear_cluster_codebook", type = int, help="set the pretrained model's cluster_codebook to empty dict. This is used when we train the segmenter on a complete new dataset. Set this to 0 if you just want to slighlt finetune the model with some additional data with the same cluster naming rule.", default = 0 )
    
    args = parser.parse_args()

    wandb.init(
        project=args.project,
        name=args.run_name,
        notes=args.run_notes,
        tags=args.run_tags,
        dir=args.wandb_dir,
    )
    wandb.define_metric("current_step")
    wandb.define_metric( "epoch", step_metric="current_step")
    wandb.define_metric( "train/loss", step_metric="current_step")
    wandb.define_metric( "train/learning_rate", step_metric="current_step")
    wandb.define_metric( "validate/score", step_metric="current_step")
    wandb.define_metric( "validate/segment_score", step_metric="current_step")
    wandb.define_metric( "validate/frame_score", step_metric="current_step")

    if args.seed is not None:
        np.random.seed(args.seed)  
        
    if args.val_ratio == 0.0:
        args.validate_every = None
        args.validate_per_epoch= None

    create_if_not_exists(args.model_folder)

    if args.gpu_list is None:
        args.gpu_list = np.arange(args.n_device).tolist()
        
    device = torch.device(  "cuda:%d"%( args.gpu_list[0] ) if torch.cuda.is_available() else "cpu" )

    model, tokenizer = load_model( args.initial_model_path, args.total_spec_columns, args.dropout) #brauche ich den tokenizer noch?

    
    
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

    if args.audio_folder and args.label_folder:
        audio_path_list_train, label_path_list_train = get_audio_and_label_paths_from_folders(
            args.audio_folder, args.label_folder)
    else:
        audio_path_list_train, label_path_list_train = get_audio_and_label_paths(args.train_dataset_folder)

    cluster_codebook = get_cluster_codebook( label_path_list_train, segmenter.cluster_codebook )
    segmenter.update_cluster_codebook( cluster_codebook )

    audio_list_train, label_list_train = load_data(audio_path_list_train, label_path_list_train, cluster_codebook = cluster_codebook, n_threads = 20 )
    
    #ToDo: heißt das ganze files werden als train val genommen?
    if args.val_ratio > 0:
        (audio_list_train, label_list_train), ( audio_list_val, label_list_val ) = train_val_split( audio_list_train, label_list_train, args.val_ratio )

    #slices audios in chunks of total_spec_columns spectogram columns and adjusts the labels accordingly
    audio_list_train, label_list_train, metadata_list = slice_audios_and_labels( audio_list_train, label_list_train, args.total_spec_columns )
    #audio_list_val, label_list_val, metadata_list = slice_audios_and_labels( audio_list_train, label_list_train, args.total_spec_columns )

    # Check if we have any data after slicing
    if len(audio_list_train) == 0:
        print("Error: No valid audio samples after slicing!")
        print("This could be due to:")
        print("  - Audio files that are too short after slicing")
        print("  - No valid segments in the labels")
        print("  - All segments being filtered out during processing")
        sys.exit(1)

    print(f"Created {len(audio_list_train)} training samples after slicing")

    training_dataset = WhisperFormerDataset( audio_list_train, label_list_train, tokenizer, args.max_length, 
                                         args.total_spec_columns)

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
    model = load_actionformer_model(args.initial_model_path, 2)
    model = nn.DataParallel( model, args.gpu_list )
    model = model.to(device)

    val_score_history = []
    esh = EarlyStopHandler(patience = args.patience)
    early_stop = False
    current_step = 0

    for epoch in range(8):  # This +1 is to ensure current_step can reach args.max_num_iterations
        print(f"\n=== Starting Epoch {epoch} ===")
        model.train() 
        training_losses = []
        val_losses = []

        for count, batch in enumerate( tqdm( training_dataloader, desc=f'epoch-{epoch:03}', disable=is_scheduled_job()) ):
            torch.set_printoptions(threshold=torch.inf)
            """sanity-check
            for i in range(4):
                tensor = batch['segments'][i,:,0]
                # Toleranz für "nahe Null"
                tol = 0.1

                # Prüfen, ob alle Werte innerhalb ±tol von 0 liegen
                if not torch.all(torch.abs(tensor) < tol):
                    print("Tensor enthält Werte, die nicht (nahe) Null sind:")
                    print(tensor)
                else:
                    print("Tensor besteht nur aus (nahe) Null-Werten.")
            """
            total_loss = actionformer_train_iteration(model, batch, optimizer, None, scaler, device)
            training_losses.append(total_loss)


    
        print(f"=== End of Epoch {epoch} ===")
        print(f"val_ratio = {args.val_ratio}, will run validation: {args.val_ratio > 0}")
        
        # Validation at the end of each epoch
        if args.val_ratio > 0:
            print(f"Running validation for epoch {epoch}...")
            
            # Create validation dataloader
            val_dataset = WhisperFormerDataset(audio_list_val, label_list_val, tokenizer, args.max_length, args.total_spec_columns)
            val_dataloader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False,
                                        num_workers=args.num_workers, collate_fn=collate_fn, drop_last=False) #eher shuffle = true oder?
            
            # Compute validation loss
            val_training_loss = actionformer_validation_loss(model, val_dataloader, device)
            
            val_losses.append(val_training_loss)
            print(f"Epoch {epoch}: Val Loss = {val_training_loss:.4f}")

            wandb.log({
                "current_step": current_step,
                "validate/training_loss": val_training_loss
            })
            
            # Update scheduler with validation loss
            scheduler.step(val_training_loss)
            print(f"Current learning rate: {get_lr(optimizer)[0]:.2e}")
        else:
            print(f"No validation set (val_ratio = {args.val_ratio})")

        if current_step >= args.max_num_iterations:
            break

    # Save final model
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    final_model_save_path = f"{args.model_folder}/final_model_{timestamp}"
    
    # Save ActionFormer model
    
    os.makedirs(final_model_save_path, exist_ok=True)
    torch.save(model.module.state_dict(), f"{final_model_save_path}/actionformer_model.pth")
    print("Training complete. Model saved to:", final_model_save_path)

    # Save training arguments
    params_path = os.path.join(final_model_save_path, "training_args.json")
    with open(params_path, "w") as f:
        json.dump(vars(args), f, indent=4)
        
    