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
from convert_hf_to_ct2 import convert_hf_to_ct2
from util.common import EarlyStopHandler
from training_utils import collate_fn, train_iteration, evaluate

def main(args):
    wandb.init(project=args.project, name=args.run_name, notes=args.run_notes, tags=args.run_tags)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    create_if_not_exists(args.model_folder)

    model, tokenizer = load_model(args.initial_model_path, args.total_spec_columns, args.dropout)
    model.to(device)
    if args.freeze_encoder:
        for p in model.model.encoder.parameters():
            p.requires_grad = False

    model = nn.DataParallel(model, args.gpu_list)
    optimizer = AdamW(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    scaler = torch.cuda.amp.GradScaler()

    segmenter = WhisperSegmenterForEval(model=model, tokenizer=tokenizer)

    #audio_paths, label_paths = get_audio_and_label_paths(args.train_dataset_folder)

    if args.audio_folder and args.label_folder:
        audio_paths, label_paths = get_audio_and_label_paths_from_folders(
            args.audio_folder, args.label_folder)
    else:
        audio_paths, label_paths = get_audio_and_label_paths(args.train_dataset_folder)

    cluster_codebook = get_cluster_codebook(label_paths, segmenter.cluster_codebook)
    segmenter.update_cluster_codebook(cluster_codebook)

    audio_list, label_list = load_data(audio_paths, label_paths, cluster_codebook, n_threads=20)

    if args.val_ratio > 0:
        (audio_list, label_list), (val_audio, val_label) = train_val_split(audio_list, label_list, args.val_ratio)

    audio_list, label_list = slice_audios_and_labels(audio_list, label_list, args.total_spec_columns)
    dataset = VocalSegDataset(audio_list, label_list, tokenizer, args.max_length,
                              args.total_spec_columns, model.module.config.species_codebook)
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True,
                            num_workers=args.num_workers, collate_fn=collate_fn, drop_last=True)
    
    if args.max_num_iterations is None:
        args.max_num_iterations = len(dataloader) * args.max_num_epochs

    #total_steps = len(dataloader) * args.max_num_epochs
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=4, min_lr=1e-10)

    current_step = 0
    training_losses = []

    for epoch in range(args.max_num_epochs):
        model.train()
        training_losses = []
        for batch in tqdm(dataloader, desc=f"Epoch {epoch}"):
            loss = train_iteration(model, batch, optimizer, scheduler, scaler, device)
            #print(loss)
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

        if args.validate_every and args.val_ratio > 0 and current_step % args.validate_every == 0:
            model.eval()
            eval_res = evaluate(val_audio, val_label, segmenter, args.batch_size, args.max_length,
                                num_trials=1, consolidation_method=None, num_beams=1)
            val_score = (eval_res["segment_wise"][-1] + eval_res["frame_wise"][-1]) * 0.5
            print(val_score)
            wandb.log({
                "current_step": current_step,
                "validate/segment_score": eval_res["segment_wise"][-1],
                "validate/frame_score": eval_res["frame_wise"][-1],
                "validate/score": val_score
            })
            scheduler.step(eval_res["segment_wise"][-1])  # Update the learning rate based on the validation score
            model.train()

        #scheduler.step(eval_res["segment_wise"][-1])  # Update the learning rate based on the validation score

        if current_step >= args.max_num_iterations:
            break

    # Save final model once after training
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    final_model_save_path = f"{args.model_folder}/final_model_{timestamp}"
    save_model(model, tokenizer, current_step, final_model_save_path, max_to_keep=1)
    print("Training complete. Model saved to:", final_model_save_path)

    # Path to your saved_model folder
    base_path = final_model_save_path

    # List all entries in the directory
    all_entries = os.listdir(base_path)

    # Filter to get only directories that start with 'checkpoint-'
    checkpoint_dirs = [d for d in all_entries if d.startswith("checkpoint-") and os.path.isdir(os.path.join(base_path, d))]

    # Optionally, sort them by number (if you want the latest or earliest one)
    checkpoint_dirs.sort(key=lambda x: int(x.split('-')[-1]))

    # Now pick the one you want (e.g., latest)
    latest_checkpoint = checkpoint_dirs[-1]  # or [0] for the earliest

    # Full path to the checkpoint
    checkpoint_path = os.path.join(base_path, latest_checkpoint)

    print(f"Using checkpoint at: {checkpoint_path}")
    convert_hf_to_ct2(model=checkpoint_path, output_dir=f"{args.model_folder}/final_checkpoint_ct2_{timestamp}", quantization="float16")
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
    parser.add_argument("--val_ratio", type=float, default=0.0)
    parser.add_argument("--max_length", type=int, default=100)
    parser.add_argument("--total_spec_columns", type=int, default=1000)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--learning_rate", type=float, default=3e-6)
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--warmup_steps", type=int, default=100)
    parser.add_argument("--dropout", type=float, default=0.0)
    parser.add_argument("--freeze_encoder", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num_workers", type=int, default=4)
    args = parser.parse_args()

    main(args)