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

from convert_hf_to_ct2 import convert_hf_to_ct2
from datautils import (VocalSegDataset, get_audio_and_label_paths, get_audio_and_label_paths_from_folders,
                       get_cluster_codebook, load_data,
                       slice_audios_and_labels, train_val_split, FIXED_CLUSTER_CODEBOOK, ID_TO_CLUSTER)
from model import WhisperSegmenterForEval, load_model, save_model
from util.common import EarlyStopHandler, is_scheduled_job
from util.confusion_framewise import confusion_matrix_framewise
from utils import *
from torch.nn.utils.rnn import pad_sequence
from whisperformer_dataset import WhisperFormerDataset
from whisperformer_model import WhisperFormer
from losses import sigmoid_focal_loss, ctr_diou_loss_1d
from torch.cuda.amp import autocast, GradScaler
from torch.nn import functional as F
from torch.optim.lr_scheduler import ReduceLROnPlateau
from transformers import WhisperFeatureExtractor

def losses(
        out_cls_logits,          # [B, T/2, C]
        out_offsets,             # [B, T/2, 2]
        gt_cls_labels,           # [B, T/2, C]
        gt_offsets,               # [B, T/2, 2]
        train_loss_weight=1,
        loss_normalizer=200,
        loss_normalizer_momentum=0.8
    ):
    # 1) Shapes check (optional, nützlich beim Debug)
    B, T, C = out_cls_logits.shape
    assert gt_cls_labels.shape == (B, T, C)
    assert out_offsets.shape == (B, T, 2)
    assert gt_offsets.shape == (B, T, 2)

    # positive positions: irgendeine Klasse vorhanden, nur um center herum, so wie in dataset!
    pos_mask = gt_offsets.sum(-1) > 0  # [B, T]

    # offsets für positive positions -> [N_pos, 2] 
    pred_offsets_pos = out_offsets[pos_mask]   # [N_pos, 2]
    gt_offsets_pos = gt_offsets[pos_mask]      # [N_pos, 2]


    # update normalizer
    num_pos = pos_mask.sum().item() 
    loss_normalizer = loss_normalizer_momentum * loss_normalizer + (
        1 - loss_normalizer_momentum
    ) * max(num_pos, 1)

    # classification targets for valid positions
    gt_target = gt_cls_labels

    logits_valid = out_cls_logits
    #print(f'logits_valid: {logits_valid}')
    #print(f'gt_target: {gt_target}')
    cls_loss = sigmoid_focal_loss(logits_valid, gt_target, reduction='sum')
    #cls_loss = cls_loss / loss_normalizer

    # regression
    if num_pos == 0:
        reg_loss = 0.0 * pred_offsets_pos.sum()
    else:
        reg_loss = ctr_diou_loss_1d(pred_offsets_pos, gt_offsets_pos, reduction='sum')
        reg_loss = reg_loss / loss_normalizer

    # balancing
    if train_loss_weight > 0:
        loss_weight = train_loss_weight
    else:
        loss_weight = cls_loss.detach() / max(reg_loss.item(), 0.01)

    final_loss = cls_loss + reg_loss * loss_weight
    return cls_loss, reg_loss, final_loss

def losses_val(
        out_cls_logits,          # [B, T/2, C]
        out_offsets,             # [B, T/2, 2]
        gt_cls_labels,           # [B, T/2, C]
        gt_offsets,               # [B, T/2, 2]
        train_loss_weight=1,
        loss_normalizer=200,
        loss_normalizer_momentum=0.8
    ):
    # 1) Shapes check (optional, nützlich beim Debug)
    B, T, C = out_cls_logits.shape
    assert gt_cls_labels.shape == (B, T, C)
    assert out_offsets.shape == (B, T, 2)
    assert gt_offsets.shape == (B, T, 2)

    # positive positions: irgendeine Klasse vorhanden, nur um center herum, so wie in dataset!
    pos_mask = gt_offsets.sum(-1) > 0  # [B, T]

    # offsets für positive positions -> [N_pos, 2] 
    pred_offsets_pos = out_offsets[pos_mask]   # [N_pos, 2]
    gt_offsets_pos = gt_offsets[pos_mask]      # [N_pos, 2]


    num_pos = pos_mask.sum().item() 


    cls_loss = sigmoid_focal_loss(out_cls_logits, gt_cls_labels, reduction='sum')/ max(num_pos,1)

    # regression
    if num_pos == 0:
        reg_loss = 0.0 * pred_offsets_pos.sum()
    else:
        reg_loss = ctr_diou_loss_1d(pred_offsets_pos, gt_offsets_pos, reduction='sum')
        reg_loss = reg_loss / num_pos

    # balancing
    if train_loss_weight > 0:
        loss_weight = train_loss_weight
    else:
        loss_weight = cls_loss.detach() / max(reg_loss.item(), 0.01)

    final_loss = cls_loss + reg_loss * loss_weight
    return cls_loss, reg_loss, final_loss

"""
def losses(out_cls, out_offsets, gt_cls, gt_offsets,
           loss_normalizer=500.0, loss_normalizer_momentum=0.9, train_loss_weight=0.0,
           valid_mask=None, use_reg=True, reg_lambda=0.5):
    # gt_cls: [B,T,C] soft targets in [0,1]
    # out_cls: [B,T,C] Logits (ohne Sigmoid)

    # 1) Loss-Normalizer (falls du Regression trainierst)
    pos_mask_time = (gt_cls.max(dim=-1).values > 0)  # [B,T]
    num_pos = pos_mask_time.sum()
    loss_normalizer = loss_normalizer_momentum * loss_normalizer + (1 - loss_normalizer_momentum) * torch.clamp(num_pos.float(), min=1.0)

    # 2) Klassifikation: BCEWithLogits mit pos_weight pro Klasse
    gt = gt_cls.float().clamp(0, 1)  # [B,T,C]
    # Optional: Loss in FP32 auswerten für Stabilität
    logits = out_cls.float()

    # Per-Batch Klassen-Statistik für pos_weight
    with torch.no_grad():
        pos = gt.sum(dim=(0, 1))                 # [C]
        neg = (1.0 - gt).sum(dim=(0, 1))         # [C]
        eps = 1e-6
        # robust: add eps num/den, clamp gegen Ausreißer
        pos_weight = ((neg + eps) / (pos + eps)).clamp(max=1e3)
        pos_weight = pos_weight.to(device=logits.device, dtype=logits.dtype)

    if valid_mask is None:
        # Reduktion: mean über alle Elemente
        cls_loss = F.binary_cross_entropy_with_logits(logits, gt, pos_weight=pos_weight, reduction='mean')
    else:
        # Gültigkeitsmaske [B,T] anwenden
        elem = F.binary_cross_entropy_with_logits(logits, gt, pos_weight=pos_weight, reduction='none')  # [B,T,C]
        elem = elem * valid_mask.unsqueeze(-1)  # [B,T,1]
        denom = (valid_mask.sum() * logits.shape[-1]).clamp_min(1)
        cls_loss = elem.sum() / denom
 
    if use_reg and pos_mask_time.any(): 
        reg_sum = ctr_diou_loss_1d(out_offsets[pos_mask_time], gt_offsets[pos_mask_time], reduction='sum') 
        denom = pos_mask_time.sum().clamp_min(1).to(reg_sum.dtype) 
        reg_loss = reg_sum / denom 
    else: reg_loss = torch.tensor(0.0, device=out_cls.device, dtype=out_cls.dtype, requires_grad=True)
    total_loss = cls_loss + (reg_lambda * reg_loss if use_reg else 0.0)

    return cls_loss, reg_loss, total_loss
"""



def load_actionformer_model(initial_model_path, num_classes):
    """Load ActionFormer model with Whisper encoder"""
    from transformers import WhisperModel
    
    # Load Whisper encoder #????this should be part of the model already!
    whisper_model = WhisperModel.from_pretrained("openai/whisper-small", local_files_only=True)
    encoder = whisper_model.encoder
    
    # Create ActionFormer model with the correct number of classes
    model = WhisperFormer(encoder, num_classes=num_classes, no_decoder = args.no_decoder)
    
    # Load pretrained weights if available
    if initial_model_path and os.path.exists(initial_model_path):
        print(f"Loading pretrained weights from {initial_model_path}")
        model.load_state_dict(torch.load(initial_model_path, map_location='cpu'))
    
    return model

# for debugging
def actionformer_train_iteration(model, batch, optimizer, scheduler, scaler, device):
    """
    Training iteration für ActionFormer mit AMP Debugging.
    Prüft Gradienten, Logits, Shapes etc. ohne das Training zu unterbrechen.
    """
    # Batch auf Device
    for key in batch:
        batch[key] = batch[key].to(device)

    optimizer.zero_grad()

    # Forward in autocast
    #with autocast(dtype=torch.float16, enabled=(device.type == "cuda")):
    with autocast(dtype=torch.float16, enabled=False):
        class_preds, regr_preds = model(batch["input_features"])
        # Compute loss
        cls_loss, reg_loss, total_loss = losses_val(
            class_preds,
            regr_preds,
            batch['clusters'],
            batch['segments'],
            loss_normalizer=200.0,
            loss_normalizer_momentum=0.8,
            train_loss_weight=args.train_loss_weight
        )
    clusters = batch['clusters']
    segments = batch['segments']


    # Backward über GradScaler
    scaler.scale(total_loss).backward()

    # Optimizer step + scaler update
    scaler.step(optimizer)
    scaler.update()

    if args.lr_schedule == "linear":
        scheduler.step()

    return float(cls_loss.detach()), float(reg_loss.detach()), float(total_loss.detach())




def actionformer_validation_loss(model, val_dataloader, device): #NEIN hier anderer loss notwendig!!
    was_training = model.training
    model.eval()

    sum_loss = 0.0
    sum_loss_class = 0.0
    sum_loss_reg = 0.0
    n_batches = 0

    with torch.no_grad():
        for batch in val_dataloader:
            for k in batch:
                batch[k] = batch[k].to(device, non_blocking=True)

            class_preds, regr_preds = model(batch["input_features"])
            cls_loss, reg_loss, total = losses_val(
                class_preds, regr_preds, batch["clusters"], batch["segments"],
                train_loss_weight=args.train_loss_weight,
                loss_normalizer=200,
                loss_normalizer_momentum=0.8
            )
            # korrekt auf CPU in Python-Float summieren
            sum_loss += float(total.detach().cpu())
            sum_loss_class += float(cls_loss.detach().cpu())
            sum_loss_reg += float(reg_loss.detach().cpu())
            n_batches += 1

    if was_training:
        model.train()

    return (sum_loss / max(n_batches, 1), sum_loss_class / max(n_batches, 1), sum_loss_reg / max(n_batches, 1))






def collate_fn(batch):
    
    input_features = [item["input_features"].clone().detach().float() for item in batch]
    segments = torch.stack([item["segments"] for item in batch])
    clusters = torch.stack([item["clusters"] for item in batch])

    input_features = torch.stack(input_features) 

    return {
        "input_features": input_features,       # [B, F, T] for B=batch_size, F=feature_dim, T=total_spec_columns
        "segments": segments,                   # [B, T/2 , 2]
        "clusters": clusters,                    # [B, T/2, C]
    }


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
    parser.add_argument("--update_every", type = int, default = 100 )
    parser.add_argument("--validate_every", type = int, default = None )
    parser.add_argument("--validate_per_epoch", type = int, default = 0 )
    parser.add_argument("--save_every", type = int, default = None )
    parser.add_argument("--save_per_epoch", type = int, default = 0 )
    parser.add_argument("--max_num_epochs", type = int, default = 10 )
    parser.add_argument("--max_num_iterations", type = int, default = None )
    parser.add_argument("--val_ratio", type = float, default = 0.1 )
    parser.add_argument("--make_equal", nargs="+", default = None )

    parser.add_argument("--patience", type = int, default = 4, help="If the validation score does not improve for [patience] epochs, stop training.")
    parser.add_argument("--total_spec_columns", type = int, default = 3000 )
    parser.add_argument("--batch_size", type = int, default = 4 )
    parser.add_argument("--learning_rate", type = float, default = 1e-4)
    parser.add_argument("--lr_schedule", default = 'plateau')
    parser.add_argument("--seed", type = int, default = 66100 )
    parser.add_argument("--weight_decay", type = float, default = 0.01 )
    parser.add_argument("--warmup_steps", type = int, default = 100 )
    parser.add_argument("--freeze_encoder", type = bool, default = True)
    parser.add_argument("--dropout", type = float, default = 0.0 )
    parser.add_argument("--num_workers", type = int, default = 4 )
    parser.add_argument("--num_classes", type = int, default = 1 )
    parser.add_argument("--scheduler_patience", type = int, default = 5)
    parser.add_argument("--factor", type = int, default = 0.3)
    parser.add_argument("--train_loss_weight", type = float, default = 0.2)
    parser.add_argument("--no_decoder", type = bool, default = False)
    parser.add_argument("--clear_cluster_codebook", type = int, help="set the pretrained model's cluster_codebook to empty dict. This is used when we train the segmenter on a complete new dataset. Set this to 0 if you just want to slighlt finetune the model with some additional data with the same cluster naming rule.", default = 0 )
    
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

    model = load_actionformer_model(args.initial_model_path, args.num_classes)
    
    if args.freeze_encoder:
        for para in model.encoder.parameters():
            para.requires_grad = False
    else:
        for para in model.encoder.parameters():
            para.requires_grad = True


    if args.audio_folder and args.label_folder:
        audio_path_list_train, label_path_list_train = get_audio_and_label_paths_from_folders(
            args.audio_folder, args.label_folder)
    else:
        audio_path_list_train, label_path_list_train = get_audio_and_label_paths(args.train_dataset_folder)

    #cluster_codebook = get_cluster_codebook( label_path_list_train, initial_cluster_codebook={}, num_classes=args.num_classes, make_equal = args.make_equal)
    cluster_codebook = FIXED_CLUSTER_CODEBOOK

    audio_list_train, label_list_train = load_data(audio_path_list_train, label_path_list_train, cluster_codebook = cluster_codebook, n_threads = 1 )
    
    if args.val_ratio > 0:
        (audio_list_train, label_list_train), ( audio_list_val, label_list_val ) = train_val_split( audio_list_train, label_list_train, args.val_ratio )

    #slices audios in chunks of total_spec_columns spectogram columns and adjusts the labels accordingly
    audio_list_train, label_list_train, metadata_list = slice_audios_and_labels( audio_list_train, label_list_train, args.total_spec_columns )
    print(f"Created {len(audio_list_train)} training samples after slicing") 

    feature_extractor = WhisperFeatureExtractor.from_pretrained("openai/whisper-small", local_files_only=True)

    if args.val_ratio > 0:
        audio_list_val, label_list_val, metadata_list = slice_audios_and_labels( audio_list_val, label_list_val, args.total_spec_columns )
        print(f"Created {len(audio_list_val)} validation samples after slicing")

        # Create validation dataloader
        val_dataset = WhisperFormerDataset(audio_list_val, label_list_val, args.total_spec_columns, feature_extractor, args.num_classes)
        val_dataloader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False,
                                        num_workers=args.num_workers, collate_fn=collate_fn, drop_last=False) 

    # Check if we have any data after slicing
    if len(audio_list_train) == 0:
        print("Error: No valid audio samples after slicing!")
        print("This could be due to:")
        print("  - Audio files that are too short after slicing")
        print("  - No valid segments in the labels")
        print("  - All segments being filtered out during processing")
        sys.exit(1)


    training_dataset = WhisperFormerDataset(audio_list_train, label_list_train, 
                                         args.total_spec_columns, feature_extractor, args.num_classes)

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
    """
    #sanity check data loader
    batch = next(iter(training_dataloader))
    x =batch['segments']
    # Indizes der nicht-null Elemente finden
    indices = torch.nonzero(x, as_tuple=False)
        # Ausgabe der Positionen und Werte
    for idx in indices:
        value = x[tuple(idx)]
        print(f"Position: {tuple(idx.tolist())}, Wert: {value.item()}")
    """

    if args.max_num_iterations is not None and args.max_num_iterations > 0:
        args.max_num_epochs = int(np.ceil( args.max_num_iterations / len( training_dataloader )  ))
    else:
        assert args.max_num_epochs is not None and args.max_num_epochs > 0
        args.max_num_iterations = len( training_dataloader ) * args.max_num_epochs
                

    #model = load_actionformer_model(args.initial_model_path, 1)
    model = nn.DataParallel( model, args.gpu_list )
    model = model.to(device)


    no_decay = ['bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [
        {'params': [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)], 'weight_decay': args.weight_decay},
        {'params': [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
    ]
    
    optimizer = AdamW(optimizer_grouped_parameters, lr = args.learning_rate )





    #TODO anderen scheduler hier wählen!
    if args.lr_schedule == "linear":
        scheduler = get_linear_schedule_with_warmup(
            optimizer, num_warmup_steps= args.warmup_steps, 
            num_training_steps = args.max_num_iterations
        )
    elif args.lr_schedule == "plateau":
        scheduler = ReduceLROnPlateau(
            optimizer,
            mode="min",               # or "max" if you monitor accuracy/score
            factor=args.factor,               # reduce LR by half
            patience=args.scheduler_patience,               # number of evals with no improvement before reducing
            verbose=True
        )
    else:
        scheduler = None
    
    scaler = torch.cuda.amp.GradScaler()

    val_score_history = []
    esh = EarlyStopHandler(patience = args.patience)
    early_stop = False
    current_step = 0

    train_loss_history = []
    train_loss_history_class=[]
    train_loss_history_reg=[]
    val_loss_history = []
    val_loss_history_class = []
    val_loss_history_reg = []
    lr_reduction_epochs = []




    for epoch in range(args.max_num_epochs+1):  
        print(f"\n=== Starting Epoch {epoch} ===")
        model.train() 

        training_losses = []
        training_losses_class = []
        training_losses_reg = []
        val_losses = []
        lrs=[]

        for count, batch in enumerate( tqdm( training_dataloader, desc=f'epoch-{epoch:03}', disable=is_scheduled_job()) ):
            cls_loss, reg_loss, total_loss = actionformer_train_iteration(model, batch, optimizer, scheduler, scaler, device)
            training_losses.append(total_loss)
            training_losses_class.append(cls_loss)
            training_losses_reg.append(reg_loss)

            if count % 100 == 0:
                print(f"Epoch {epoch}, Step {count}, Training Total Loss: {total_loss:.4f}")
                print(f"Epoch {epoch}, Step {count}, Training Class Loss: {cls_loss:.4f}")
                print(f"Epoch {epoch}, Step {count}, Training Regression Loss: {reg_loss:.4f}")
                

        print(f"=== End of Epoch {epoch} ===")
        epoch_train_loss = sum(training_losses)/len(training_losses)
        epoch_train_loss_class = sum(training_losses_class)/len(training_losses_class)
        epoch_train_loss_reg = sum(training_losses_reg)/len(training_losses_reg)
        train_loss_history.append(epoch_train_loss)
        train_loss_history_class.append(epoch_train_loss_class)
        train_loss_history_reg.append(epoch_train_loss_reg)
        print(f"Epoch {epoch}, Step {count}, Epoch Training Loss: {epoch_train_loss:.4f}")
        print(f"val_ratio = {args.val_ratio}, will run validation: {args.val_ratio > 0}")



        # Validation at the end of each epoch
        if args.val_ratio > 0:
            print(f"Running validation for epoch {epoch}...")
            
            # Compute validation loss
            val_loss, val_loss_class, val_loss_reg = actionformer_validation_loss(model, val_dataloader, device)

            if args.lr_schedule == "plateau":
                old_lr = get_lr(optimizer)[0]
                scheduler.step(val_loss)
                new_lr = get_lr(optimizer)[0]
                if new_lr < old_lr:
                    print(f"LR reduced from {old_lr:.2e} to {new_lr:.2e} at epoch {epoch}")
                    lr_reduction_epochs.append((epoch, new_lr))
            
            
            val_loss_history.append(val_loss)
            val_loss_history_class.append(val_loss_class)
            val_loss_history_reg.append(val_loss_reg)
            print(f"Epoch {epoch}: Val Loss = {val_loss:.4f}")

            print(f"Current learning rate: {get_lr(optimizer)[0]:.2e}")
            lr = get_lr(optimizer)[0]
            lrs.append(lr)
        else:
            print(f"No validation set (val_ratio = {args.val_ratio})")

        if current_step >= args.max_num_epochs:
            break

    # Save final model
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    final_model_save_path = f"{args.model_folder}/final_model_{timestamp}"
    
    # Save ActionFormer model
    
    os.makedirs(final_model_save_path, exist_ok=True)

    #plot the losses
        # === Plot Loss Curves ===
    plt.figure(figsize=(8, 5))
    plt.plot(train_loss_history, label="Training Loss")
    plt.plot(train_loss_history_class, label="Training Loss Class")
    plt.plot(train_loss_history_reg, label="Training Loss Regression")
    if len(val_loss_history) > 0:
        plt.plot(val_loss_history, label="Validation Loss")
        plt.plot(val_loss_history_class, label="Validation Loss Class")
        plt.plot(val_loss_history_reg, label="Validation Loss Regression")
    
    # Vertikale Linien für LR-Reduktionen
    for epoch_idx, lr_val in lr_reduction_epochs:
        plt.axvline(x=epoch_idx, color="k", linestyle="--", alpha=0.7)
        plt.text(epoch_idx, max(train_loss_history), f"LR={lr_val:.1e}",
                rotation=90, verticalalignment="bottom", fontsize=8, color="k")

    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training and Validation Loss")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()

    # Save plot
    loss_plot_path = os.path.join(final_model_save_path, "loss_curve.png")
    plt.savefig(loss_plot_path)
    print("Saved loss curve to:", loss_plot_path)
    plt.close()


    torch.save(model.module.state_dict(), f"{final_model_save_path}/actionformer_model.pth")
    print("Training complete. Model saved to:", final_model_save_path)

    # Save training arguments
    params_path = os.path.join(final_model_save_path, "training_args.json")
    with open(params_path, "w") as f:
        json.dump(vars(args), f, indent=4)
        
