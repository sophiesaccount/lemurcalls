import argparse
import os
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim import AdamW
from transformers import get_linear_schedule_with_warmup
from tqdm import tqdm
import wandb

from utils import get_lr, create_if_not_exists
from datautils import (VocalSegDataset, get_audio_and_label_paths,
                       get_cluster_codebook, load_data, slice_audios_and_labels, train_val_split)
from model import WhisperSegmenterForEval, load_model, save_model
from convert_hf_to_ct2 import convert_hf_to_ct2
from util.common import EarlyStopHandler
from training_utils import collate_fn, train_iteration, evaluate  # Move those helper functions to training_utils.py

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

    total_steps = len(dataloader) * args.max_num_epochs
    scheduler = get_linear_schedule_with_warmup(optimizer, args.warmup_steps, total_steps)

    esh = EarlyStopHandler(patience=args.patience)
    current_step = 0
    training_losses = []
    val_score_history = []

    for epoch in range(args.max_num_epochs):
        model.train()
        for batch in tqdm(dataloader, desc=f"Epoch {epoch}"):
            loss = train_iteration(model, batch, optimizer, scheduler, scaler, device)
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
            wandb.log({
                "current_step": current_step,
                "validate/segment_score": eval_res["segment_wise"][-1],
                "validate/frame_score": eval_res["frame_wise"][-1],
                "validate/score": val_score
            })
            val_score_history.append((current_step, val_score))
            if esh and esh.check(val_score):
                break
            save_model(model, tokenizer, current_step, args.model_folder, args.max_to_keep)
            model.train()

        if args.save_every and current_step % args.save_every == 0:
            save_model(model, tokenizer, current_step, args.model_folder, args.max_to_keep)

                # Force one validation at end of epoch if not yet done
        if args.val_ratio > 0:
            model.eval()
            eval_res = evaluate(val_audio, val_label, segmenter, args.batch_size, args.max_length,
                                num_trials=1, consolidation_method=None, num_beams=1)
            val_score = (eval_res["segment_wise"][-1] + eval_res["frame_wise"][-1]) * 0.5
            wandb.log({
                "current_step": current_step,
                "validate/segment_score": eval_res["segment_wise"][-1],
                "validate/frame_score": eval_res["frame_wise"][-1],
                "validate/score": val_score
            })
            val_score_history.append((current_step, val_score))
            if esh and esh.check(val_score):
                break
            save_model(model, tokenizer, current_step, args.model_folder, args.max_to_keep)
            model.train()

        if current_step >= args.max_num_iterations:
            break

    if val_score_history:
        best_ckpt = sorted(val_score_history, key=lambda x: -x[1])[0][0]
        best_model_path = f"{args.model_folder}/checkpoint-{best_ckpt}"
        final_ckpt_path = f"{args.model_folder}/final_checkpoint"

        if os.path.exists(best_model_path):
            os.system(f"cp -r {best_model_path} {final_ckpt_path}")
            os.system(f"rm -r {args.model_folder}/checkpoint-*")
        else:
            print(f"Warning: Best checkpoint not found at {best_model_path}, skipping final checkpoint setup.")
            return
    else:
        print("No validation score recorded; no final checkpoint to convert.")
        return

    convert_hf_to_ct2(model=f"{args.model_folder}/final_checkpoint", output_dir=f"{args.model_folder}/final_checkpoint_ct2", quantization="float16")

    print("Training complete.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--initial_model_path")
    parser.add_argument("--model_folder")
    parser.add_argument("--train_dataset_folder")
    parser.add_argument("--n_device", type=int, default=1)
    parser.add_argument("--gpu_list", type=int, nargs="+", default=[0])
    parser.add_argument("--project", default="wseg-lemur")
    parser.add_argument("--run_name", default=None)
    parser.add_argument("--run_notes", default=None)
    parser.add_argument("--run_tags", nargs='+', default=None)
    parser.add_argument("--update_every", type=int, default=100)
    parser.add_argument("--validate_every", type=int, default=50)
    parser.add_argument("--save_every", type=int, default=1000)
    parser.add_argument("--max_num_epochs", type=int, default=3)
    parser.add_argument("--max_num_iterations", type=int, default=None)
    parser.add_argument("--val_ratio", type=float, default=0.0)
    parser.add_argument("--patience", type=int, default=10)
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
    parser.add_argument("--max_to_keep", type=int, default=3)
    args = parser.parse_args()

    main(args)
