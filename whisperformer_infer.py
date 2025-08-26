import argparse
import json
import os
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from whisperformer_dataset import WhisperFormerDataset
from whisperformer_model import WhisperFormer
from transformers import WhisperModel
from datautils import get_audio_and_label_paths_from_folders, load_data, get_cluster_codebook
from datautils import slice_audios_and_labels
from whisperformer_train import collate_fn  # Reuse collate function from training
import numpy as np
from collections import defaultdict

def load_trained_whisperformer(checkpoint_path, num_classes, device):
    """Load the WhisperFormer model with Whisper encoder and trained weights."""
    whisper_model = WhisperModel.from_pretrained("openai/whisper-tiny")
    encoder = whisper_model.encoder

    model = WhisperFormer(encoder, num_classes=num_classes)
    model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    model.to(device)
    model.eval()
    return model



def nms_1d_torch(intervals: torch.Tensor, iou_threshold: float = 0.5):
    """
    intervals: Tensor [N, 3] -> (start, end, score)
    iou_threshold: IoU Threshold for suppression

    returns: Tensor [M, 3] of kept intervals
    """
    if intervals.numel() == 0:
        return intervals.new_zeros((0, 3))

    starts = intervals[:, 0]
    ends = intervals[:, 1]
    scores = intervals[:, 2]

    # Sort by score descending
    order = scores.argsort(descending=True)
    keep = []

    while order.numel() > 0:
        i = order[0]
        keep.append(i)

        if order.numel() == 1:
            break

        # compute IoU with the rest
        ss = torch.maximum(starts[i], starts[order[1:]])
        ee = torch.minimum(ends[i], ends[order[1:]])
        inter = torch.clamp(ee - ss, min=0)

        union = (ends[i] - starts[i]) + (ends[order[1:]] - starts[order[1:]]) - inter
        iou = inter / union

        # keep only intervals with IoU <= threshold
        inds = (iou <= iou_threshold).nonzero(as_tuple=False).squeeze(1)
        order = order[inds + 1]

    out = intervals[keep]
    if out.ndim == 1:   # turn single interval into [1,3]
        out = out.unsqueeze(0)
    return out



def run_inference_new(model, dataloader, device, threshold):
    iou_threshold = 0.5
    all_preds = []
    min_duration = 0.001
    sec_per_col = 0.0025

    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Inference"):
            for key in batch:
                batch[key] = batch[key].to(device)
            
            with torch.amp.autocast(device_type="cuda", dtype=torch.float16):
                class_preds, regr_preds = model(batch["input_features"])
                class_probs = torch.sigmoid(class_preds)

                B, T, _ = regr_preds.shape[:3]
                _, _, C = class_probs.shape

                for b in range(B):
                    for c in range(C):
                        intervals = []
                        for t in range(T):
                            score = class_probs[b, t, c]
                            if score > threshold:
                                start = t - regr_preds[b, t, c, 0]
                                end   = t + regr_preds[b, t, c, 1]
                                interval = torch.stack([start, end, score])
                                #interval = torch.round(interval * 100) / 100
                                intervals.append(interval)

                                # only keep intervals longer than min_duration
                                #duration_sec = (interval[1] - interval[0]) * sec_per_col
                                #if duration_sec >= min_duration:
                                #    intervals.append(interval)

                        if len(intervals) > 0:
                            intervals = torch.stack(intervals)  # [N, 3]
                            kept = nms_1d_torch(intervals, iou_threshold=iou_threshold)
                            kept = kept.cpu().tolist()
                        else:
                            kept = []

                        all_preds.append({
                            "batch": b,
                            "class": c,
                            "intervals": kept
                        })
    return all_preds




 
        

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint_path", required=True, help="Path to the .pth trained model")
    parser.add_argument("--audio_folder", required=True)
    parser.add_argument("--label_folder", required=True)  
    parser.add_argument("--output_json", default="inference_results.json")
    parser.add_argument("--max_length", type=int, default=100)
    parser.add_argument("--total_spec_columns", type=int, default=3000)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--num_classes", type=int, default=2)
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    # ===== Data loading =====
    audio_paths, label_paths = get_audio_and_label_paths_from_folders(args.audio_folder, args.label_folder)
    
    # Create a dummy cluster codebook to use load_data
    cluster_codebook = get_cluster_codebook(label_paths, {})
    
    # Load audio + labels
    audio_list, label_list = load_data(audio_paths, label_paths, cluster_codebook=cluster_codebook, n_threads=1)
    
    # Slice to fit model spec length
    audio_list, label_list, metadata_list = slice_audios_and_labels(audio_list, label_list, args.total_spec_columns)

    dataset = WhisperFormerDataset(audio_list, label_list, tokenizer=None, 
                                   max_length=args.max_length, total_spec_columns=args.total_spec_columns)
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, 
                            collate_fn=collate_fn, drop_last=False)

    # ===== Model loading =====
    model = load_trained_whisperformer(args.checkpoint_path, args.num_classes, args.device)

    # ===== Inference =====
    all_preds = run_inference_new(model, dataloader, args.device, args.threshold)



    # Map predictions zurück zu den Original-Audios
    grouped_preds = defaultdict(list)

    for pred, meta in zip(all_preds, metadata_list):
        grouped_preds[meta["original_idx"]].append({
            "segment_idx": meta["segment_idx"],
            "class": pred["class"],
            "intervals": pred["intervals"]  
        })

    # ===== Reconstruction =====

    sec_per_col = 0.0025
    final_preds = {}

    for orig_idx, segs in grouped_preds.items():
        segs_sorted = sorted(segs, key=lambda x: x["segment_idx"])  

        classes, onsets, offsets = [], [], []

        for seg in segs_sorted:
            offset_cols = seg["segment_idx"] * args.total_spec_columns
            for (start_col, end_col, score) in seg["intervals"]:  # jetzt schon Python-Liste
                start_sec = (offset_cols + start_col) * sec_per_col
                end_sec   = (offset_cols + end_col)   * sec_per_col

                classes.append(seg["class"])
                onsets.append(round(float(start_sec), 3))
                offsets.append(round(float(end_sec), 3))

        final_preds[orig_idx] = {
            "onset": onsets,
            "offset": offsets,
            "cluster": classes
        }
        
    total = sum(len(preds["cluster"]) for preds in final_preds.values())
    print(total)


    # ===== Save as .json =====
    with open(args.output_json, "w") as f:
        json.dump(final_preds, f, indent=2)

    print(f"Inference complete. Results saved to {args.output_json}")


# Es fehlt: Überlappungen und Majority Voting!