import argparse
import os
import shutil
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim import AdamW
from transformers import get_linear_schedule_with_warmup
from tqdm import tqdm
import wandb
from datetime import datetime
import json

from utils import get_lr, create_if_not_exists
from datautils import (VocalSegDataset, get_audio_and_label_paths, get_audio_and_label_paths_from_folders,
                       get_cluster_codebook, load_data, slice_audios_and_labels, train_val_split)
from model import WhisperSegmenterForEval, load_model, save_model
from whisperformer_model import WhisperFormer 
from whisperformer_dataset import WhisperFormerDataset
from convert_hf_to_ct2 import convert_hf_to_ct2
from util.common import EarlyStopHandler
from training_utils import collate_fn, train_iteration, evaluate


def load_actionformer_model(initial_model_path, num_classes):
    """Load ActionFormer model with Whisper encoder"""
    from transformers import WhisperModel
    
    # Load Whisper encoder #????this should be part of the model already!
    whisper_model = WhisperModel.from_pretrained("openai/whisper-small")
    encoder = whisper_model.encoder
    
    # Create ActionFormer model with the correct number of classes
    model = WhisperFormer(encoder, num_classes=num_classes)
    
    # Load pretrained weights if available
    if initial_model_path and os.path.exists(initial_model_path):
        print(f"Loading pretrained weights from {initial_model_path}")
        model.load_state_dict(torch.load(initial_model_path, map_location='cpu'))
    
    return model


def actionformer_collate_fn(batch):
    """Custom collate function for ActionFormer model"""
    # Extract input features and labels
    input_features = [item["input_features"].clone().detach().float() for item in batch]
    labels = [item["labels"].clone().detach().long() for item in batch]
    
    # Pad sequences to same length
    input_features = torch.nn.utils.rnn.pad_sequence(input_features, batch_first=True)
    labels = torch.nn.utils.rnn.pad_sequence(labels, batch_first=True)
    
    return {
        "input_features": input_features,
        "labels": labels,
    }

#TODO
def losses(
        self, fpn_masks,
        out_cls_logits, out_offsets,
        gt_cls_labels, gt_offsets
    ):
        # fpn_masks, out_*: F (List) [B, T_i, C]
        # gt_* : B (list) [F T, C]
        # fpn_masks -> (B, FT)
        valid_mask = torch.cat(fpn_masks, dim=1)

        # 1. classification loss
        # stack the list -> (B, FT) -> (# Valid, )
        gt_cls = torch.stack(gt_cls_labels)
        pos_mask = torch.logical_and((gt_cls.sum(-1) > 0), valid_mask)

        # cat the predicted offsets -> (B, FT, 2 (xC)) -> # (#Pos, 2 (xC))
        pred_offsets = torch.cat(out_offsets, dim=1)[pos_mask]
        gt_offsets = torch.stack(gt_offsets)[pos_mask]

        # update the loss normalizer
        num_pos = pos_mask.sum().item()
        self.loss_normalizer = self.loss_normalizer_momentum * self.loss_normalizer + (
            1 - self.loss_normalizer_momentum
        ) * max(num_pos, 1)

        # gt_cls is already one hot encoded now, simply masking out
        gt_target = gt_cls[valid_mask]

        # optinal label smoothing
        gt_target *= 1 - self.train_label_smoothing
        gt_target += self.train_label_smoothing / (self.num_classes + 1)

        # focal loss
        cls_loss = sigmoid_focal_loss(
            torch.cat(out_cls_logits, dim=1)[valid_mask],
            gt_target,
            reduction='sum'
        )
        cls_loss /= self.loss_normalizer

        # 2. regression using IoU/GIoU loss (defined on positive samples)
        if num_pos == 0:
            reg_loss = 0 * pred_offsets.sum()
        else:
            # giou loss defined on positive samples
            reg_loss = ctr_diou_loss_1d(
                pred_offsets,
                gt_offsets,
                reduction='sum'
            )
            reg_loss /= self.loss_normalizer

        if self.train_loss_weight > 0:
            loss_weight = self.train_loss_weight
        else:
            loss_weight = cls_loss.detach() / max(reg_loss.item(), 0.01)

        # return a dict of losses
        final_loss = cls_loss + reg_loss * loss_weight
        return {'cls_loss'   : cls_loss,
                'reg_loss'   : reg_loss,
                'final_loss' : final_loss}

def actionformer_train_iteration(model, batch, optimizer, scheduler, scaler, device):
    """Training iteration for ActionFormer model"""
    for key in batch:
        batch[key] = batch[key].to(device)
    
    optimizer.zero_grad()
    
    with torch.amp.autocast(device_type="cuda", dtype=torch.float16):
        # Forward pass
        class_preds, regr_preds = model(batch["input_features"])
        
        # Calculate losses
        cls_loss, reg_loss, total_loss = losses(fpn_masks, out_cls_logits, out_offsets, gt_cls_labels, gt_offsets)

    
    scaler.scale(total_loss).backward()
    scaler.step(optimizer)
    scaler.update()
    
    return total_loss.item()


def actionformer_validation_loss(model, val_dataloader, device):
    """Compute validation loss for ActionFormer model"""
    model.eval()
    total_loss = 0.0
    batch_count = 0
    
    with torch.no_grad():
        for batch in val_dataloader:
            for key in batch:
                batch[key] = batch[key].to(device)
            
            with torch.amp.autocast(device_type="cuda", dtype=torch.float16):
                class_preds, regr_preds = model(batch["input_features"])
                
                # Calculate losses (same as training) #TODO
                class_loss = nn.CrossEntropyLoss()(class_preds.view(-1, class_preds.size(-1)), 
                                                  batch["labels"].view(-1))
                regr_loss = nn.MSELoss()(regr_preds, torch.zeros_like(regr_preds))  # Placeholder
                
                total_loss += (class_loss + 0.1 * regr_loss).item()
                batch_count += 1
    
    model.train()
    return total_loss / batch_count if batch_count > 0 else 0.0


def main(args):
    wandb.init(project=args.project, name=args.run_name, notes=args.run_notes, tags=args.run_tags)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    create_if_not_exists(args.model_folder)

    # Load data first to determine num_classes
    if args.audio_folder and args.label_folder:
        audio_paths, label_paths = get_audio_and_label_paths_from_folders(
            args.audio_folder, args.label_folder)
    else:
        audio_paths, label_paths = get_audio_and_label_paths(args.train_dataset_folder)

    # Create a proper cluster_codebook by analyzing the labels first
    print("Creating cluster codebook from labels...")
    cluster_codebook = get_cluster_codebook(label_paths, {})
    print(f"Found clusters: {list(cluster_codebook.keys())}")
    
    # Load and process data
    audio_list, label_list = load_data(audio_paths, label_paths, cluster_codebook=cluster_codebook, n_threads=20)
    
    print(f"Loaded {len(audio_list)} audio samples and {len(label_list)} label samples")
    
    if len(audio_list) == 0:
        print("ERROR: No audio samples loaded! Check your data paths.")
        return

    if args.val_ratio > 0:
        (audio_list, label_list), (val_audio, val_label) = train_val_split(audio_list, label_list, args.val_ratio)
        print(f"Created validation set with {len(val_audio)} samples (val_ratio = {args.val_ratio})")
    else:
        print(f"No validation set created (val_ratio = {args.val_ratio})")

    # Create dataset to determine num_classes
    dataset = ActionFormerDataset(audio_list, label_list, args.max_length, args.total_spec_columns)
    num_classes = dataset.num_classes
    print(f"Using {num_classes} classes for ActionFormer model")
    
    if len(dataset) == 0:
        print("ERROR: Dataset has 0 samples! Check your data processing.")
        return

    # Now load the model with the correct num_classes
    model = load_actionformer_model(args.initial_model_path, num_classes)
    model.to(device)
    
    if args.freeze_encoder:
        for p in model.encoder.parameters():
            p.requires_grad = False

    model = nn.DataParallel(model, args.gpu_list)
    optimizer = AdamW(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    scaler = torch.cuda.amp.GradScaler()

    # Create dataloader
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True,
                            num_workers=args.num_workers, collate_fn=actionformer_collate_fn, drop_last=True)

    if args.max_num_iterations is None:
        args.max_num_iterations = len(dataloader) * args.max_num_epochs

    print(f"Training for {args.max_num_epochs} epochs with val_ratio = {args.val_ratio}")
    print(f"Validation will run: {'YES' if args.val_ratio > 0 else 'NO'}")

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=5, min_lr=1e-12)

    current_step = 0
    training_losses = []
    val_losses = []

    for epoch in range(args.max_num_epochs):
        print(f"\n=== Starting Epoch {epoch} ===")
        model.train()
        training_losses = []
        
        for batch in tqdm(dataloader, desc=f"Epoch {epoch}"):
            loss = actionformer_train_iteration(model, batch, optimizer, None, scaler, device)
            training_losses.append(loss)
            current_step += 1

            if current_step % args.update_every == 0:
                wandb.log({
                    "current_step": current_step,
                    "train/loss": np.mean(training_losses),
                    "train/learning_rate": get_lr(optimizer)[0],
                    "epoch": epoch
                })
                training_losses.clear()

            if current_step >= args.max_num_iterations:
                break

        print(f"=== End of Epoch {epoch} ===")
        print(f"val_ratio = {args.val_ratio}, will run validation: {args.val_ratio > 0}")
        
        # Validation at the end of each epoch
        if args.val_ratio > 0:
            print(f"Running validation for epoch {epoch}...")
            
            # Create validation dataloader
            val_dataset = ActionFormerDataset(val_audio, val_label, args.max_length, args.total_spec_columns)
            val_dataloader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False,
                                        num_workers=args.num_workers, collate_fn=actionformer_collate_fn, drop_last=False)
            
            # Compute validation loss
            val_training_loss = actionformer_validation_loss(model, val_dataloader, device)
            
            val_losses.append(val_training_loss)
            print(f"Epoch {epoch}: Training-style Val Loss = {val_training_loss:.4f}")

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
    torch.save(model.module.state_dict(), f"{final_model_save_path}/actionformer_model.pth")
    print("Training complete. Model saved to:", final_model_save_path)

    # Save training arguments
    params_path = os.path.join(final_model_save_path, "training_args.json")
    with open(params_path, "w") as f:
        json.dump(vars(args), f, indent=4)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--initial_model_path")
    parser.add_argument("--model_folder")
    parser.add_argument("--audio_folder")
    parser.add_argument("--label_folder")
    parser.add_argument("--train_dataset_folder", default=None)
    parser.add_argument("--n_device", type=int, default=1)
    parser.add_argument("--gpu_list", type=int, nargs="+", default=[0])
    parser.add_argument("--project", default="wseg-lemur")
    parser.add_argument("--run_name", default=None)
    parser.add_argument("--run_notes", default=None)
    parser.add_argument("--run_tags", nargs='+', default=None)
    parser.add_argument("--update_every", type=int, default=100)
    parser.add_argument("--validate_every", type=int, default=50)
    parser.add_argument("--max_num_epochs", type=int, default=3)
    parser.add_argument("--max_num_iterations", type=int, default=None)
    parser.add_argument("--val_ratio", type=float, default=0.1)
    parser.add_argument("--max_length", type=int, default=100)
    parser.add_argument("--total_spec_columns", type=int, default=1000)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--learning_rate", type=float, default=5e-5)
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--warmup_steps", type=int, default=100)
    parser.add_argument("--dropout", type=float, default=0.0)
    parser.add_argument("--freeze_encoder", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num_workers", type=int, default=4)
    args = parser.parse_args()

    main(args)