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
    #print("intervals.shape:", intervals.shape)
    #print("keep.shape:", keep.shape, keep.dtype)
    #out = intervals[keep]
    out = intervals[torch.tensor(keep, dtype=torch.long, device=intervals.device)]
    if out.ndim == 1:   # turn single interval into [1,3]
        out = out.unsqueeze(0)
    return out



def run_inference_new(model, dataloader, device, threshold):
    iou_threshold = 0.5
    all_preds = []
    min_duration = 0.001
    sec_per_col = 0.005

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
                                start = t - regr_preds[b, t, 0]
                                end   = t + regr_preds[b, t, 1]
                                interval = torch.stack([start, end, score])
                                intervals.append(interval)

                                # only keep intervals longer than min_duration
                                duration_sec = (interval[1] - interval[0]) * sec_per_col
                                if duration_sec >= min_duration:
                                    intervals.append(interval)

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

def make_trials(audio_list, label_list, total_spec_columns, num_trials=1):
    """Erzeuge mehrere überlappende Trials (z. B. mit 1/3 Versatz)."""
    all_audio, all_label, all_meta = [], [], []
    hop = total_spec_columns // num_trials
    for trial in range(num_trials):
        audios, labels, metas = slice_audios_and_labels(
            audio_list, label_list, total_spec_columns, offset=trial*hop
        )
        for m in metas:
            m["trial"] = trial
        all_audio.extend(audios)
        all_label.extend(labels)
        all_meta.extend(metas)
    return all_audio, all_label, all_meta


def consolidate_trials_by_voting(preds, min_votes=2, iou_threshold=0.3):
    """
    preds: Liste [(start_sec, end_sec, class, trial), ...] für eine Originalaufnahme
    min_votes: wie viele Trials müssen zustimmen?
    iou_threshold: wann gelten zwei Calls als derselbe?
    """
    final_preds = []
    used = [False] * len(preds)

    for i, (s1, e1, c1, t1) in enumerate(preds):
        if used[i]:
            continue
        votes = 1
        overlaps = [(s1, e1, c1)]
        for j, (s2, e2, c2, t2) in enumerate(preds):
            if i == j or used[j]:
                continue
            if c1 != c2:
                continue
            # IoU berechnen
            inter = max(0, min(e1, e2) - max(s1, s2))
            union = (e1 - s1) + (e2 - s2) - inter
            iou = inter / union if union > 0 else 0
            if iou > iou_threshold:
                votes += 1
                overlaps.append((s2, e2, c2))
                used[j] = True

        if votes >= min_votes:
            # Mittelwert der überlappenden Intervalle nehmen
            start = np.mean([s for (s, e, c) in overlaps])
            end   = np.mean([e for (s, e, c) in overlaps])
            final_preds.append((start, end, c1))

    return final_preds


def majority_voting_across_trials(final_preds, iou_threshold=0.3):
    """
    Filter predictions via majority voting across trials.
    A call is only kept if it overlaps with a call from at least one other trial.

    final_preds: dict[orig_idx] -> dict with onset/offset/cluster
                 (must include 'offset_frac' in metadata)
    """
    voted_preds = {}

    for orig_idx, pred in final_preds.items():
        onsets = np.array(pred["onset"])
        offsets = np.array(pred["offset"])
        clusters = np.array(pred["cluster"])
        fracs = np.array(pred["offset_frac"])  # kommt aus metadata beim Reconstruction-Schritt!

        keep_mask = np.zeros(len(onsets), dtype=bool)

        for i in range(len(onsets)):
            this_on, this_off, this_frac = onsets[i], offsets[i], fracs[i]

            # IoU mit allen anderen Trials
            overlaps = []
            for j in range(len(onsets)):
                if i == j:
                    continue
                if fracs[j] == this_frac:  # nur andere Trials zählen
                    continue

                inter = max(0, min(offsets[j], this_off) - max(onsets[j], this_on))
                union = (this_off - this_on) + (offsets[j] - onsets[j]) - inter
                iou = inter / union if union > 0 else 0

                overlaps.append(iou > iou_threshold)

            # Call behalten, wenn er in >=1 anderem Trial überschneidet
            if any(overlaps):
                keep_mask[i] = True

        voted_preds[orig_idx] = {
            "onset": onsets[keep_mask].tolist(),
            "offset": offsets[keep_mask].tolist(),
            "cluster": clusters[keep_mask].tolist()
        }

    return voted_preds


def reconstruct_predictions(all_preds, metadata_list, total_spec_columns, sec_per_col=0.0025):
    """
    Map model predictions back to original audio timeline (in seconds).
    Keeps track of offset_frac (trial ID).
    """
    from collections import defaultdict
    grouped_preds = defaultdict(list)

    # Gruppiere nach Original-Audio
    for pred, meta in zip(all_preds, metadata_list):
        grouped_preds[meta["original_idx"]].append({
            "segment_idx": meta["segment_idx"],
            "class": pred["class"],
            "intervals": pred["intervals"],
            "offset_frac": meta["offset_frac"],
            "trial_id": meta["trial_id"]
        })

    final_preds = {}

    for orig_idx, segs in grouped_preds.items():
        segs_sorted = sorted(segs, key=lambda x: x["segment_idx"])  

        classes, onsets, offsets, fracs, ids = [], [], [], [], []

        for seg in segs_sorted:
            offset_cols = seg["segment_idx"] * args.total_spec_columns
            offset_cols += int(seg["offset_frac"] * args.total_spec_columns)  # Linkspadding-Offset

            for (start_col, end_col, score) in seg["intervals"]:  
                start_sec = (offset_cols + start_col) * sec_per_col
                end_sec   = (offset_cols + end_col)   * sec_per_col

                classes.append(seg["class"])
                onsets.append(float(start_sec))
                offsets.append(float(end_sec))
                fracs.append(seg["offset_frac"])   # Trial speichern
                ids.append(seg['trial_id'])

        final_preds[orig_idx] = {
            "onset": onsets,
            "offset": offsets,
            "cluster": classes,
            "offset_frac": fracs,
            "trial_id": ids
        }

    return final_preds



def majority_voting_across_trials(final_preds, overlap_threshold=0.001):
    """
    final_preds: dict[original_idx] -> {"onset":[], "offset":[], "cluster":[]}
    overlap_threshold: minimale Überlappung in Sekunden, um ein Event als bestätigt zu werten
    """
    voted_preds = {}
    
    for orig_idx, preds in final_preds.items():
        onsets = np.array(preds["onset"])
        offsets = np.array(preds["offset"])
        clusters = np.array(preds["cluster"])
        keep = np.zeros(len(onsets), dtype=bool)
        
        for i in range(len(onsets)):
            for j in range(len(onsets)):
                if i == j:
                    continue
                # Überlappung prüfen
                overlap = max(0, min(offsets[i], offsets[j]) - max(onsets[i], onsets[j]))
                if overlap >= overlap_threshold:
                    keep[i] = True
                    break
        
        voted_preds[orig_idx] = {
            "onset": list(onsets[keep]),
            "offset": list(offsets[keep]),
            "cluster": list(clusters[keep])
        }
    
    return voted_preds

def majority_voting_across_trials_efficient(final_preds, overlap_threshold=0.0001):
    """
    Majority voting across trials (efficient version).
    Only compares events that are likely to overlap.
    """
    voted_preds = {}

    for orig_idx, preds in final_preds.items():
        onsets = np.array(preds["onset"])
        offsets = np.array(preds["offset"])
        clusters = np.array(preds["cluster"])
        fracs = np.array(preds["offset_frac"])
        ids = np.array(preds["trial_id"], dtype=int)
        
        N = len(onsets)
        keep = np.zeros(N, dtype=bool)
        
        # Sortiere nach Startzeit
        order = np.argsort(onsets)
        onsets_s = onsets[order]
        offsets_s = offsets[order]
        clusters_s = clusters[order]
        fracs_s = fracs[order]
        ids_s =ids[order]
        
        for i in range(N):
            this_on = onsets_s[i]
            this_off = offsets_s[i]
            this_frac = fracs_s[i]
            this_class = clusters_s[i]
            this_id = ids_s[i]
            
            # Betrachte nur Events, die starten bevor dieses endet + overlap_threshold
            j_start = i + 1
            while j_start < N and onsets_s[j_start] <= this_off + overlap_threshold:
                if ids_s[j_start] != this_id and clusters_s[j_start] == this_class:
                    # Überlappung prüfen
                    overlap = max(0, min(this_off, offsets_s[j_start]) - max(this_on, onsets_s[j_start]))
                    if overlap >= overlap_threshold:
                        keep[order[i]] = True
                        keep[order[j_start]] = True
                j_start += 1
        
        voted_preds[orig_idx] = {
            "onset": list(onsets[keep]),
            "offset": list(offsets[keep]),
            "cluster": list(clusters[keep])
        }

    return voted_preds

        

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint_path", required=True, help="Path to the .pth trained model")
    parser.add_argument("--audio_folder", required=True)
    parser.add_argument("--label_folder", required=True)  
    parser.add_argument("--output_json", default="inference_results.json")
    parser.add_argument("--max_length", type=int, default=100)
    parser.add_argument("--total_spec_columns", type=int, default=3000)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--num_classes", type=int, default=1)
    parser.add_argument("--threshold", type=float, default=0.2)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    # ===== Data loading =====
    audio_paths, label_paths = get_audio_and_label_paths_from_folders(args.audio_folder, args.label_folder)
    
    # Create a dummy cluster codebook to use load_data
    cluster_codebook = get_cluster_codebook(label_paths, {})
    
    # Load audio + labels
    audio_list, label_list = load_data(audio_paths, label_paths, cluster_codebook=cluster_codebook, n_threads=1)
    
    # Slice to fit model spec length, slices three times each with 1/3 offset shifted
    audio_list, label_list, metadata_list = slice_audios_and_labels(audio_list, label_list, args.total_spec_columns, num_trials=1)

    # Load data
    dataset = WhisperFormerDataset(audio_list, label_list, tokenizer=None, 
                                   max_length=args.max_length, total_spec_columns=args.total_spec_columns)
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, 
                            collate_fn=collate_fn, drop_last=False)

    # ===== Model loading =====
    model = load_trained_whisperformer(args.checkpoint_path, args.num_classes, args.device)

    # ===== Inference ===== (per slice)
    print('starting inference...')
    all_preds = run_inference_new(model, dataloader, args.device, args.threshold)

    # ===== Reconstruction =====
    print('starting reconstruction...')
    final_preds = reconstruct_predictions(all_preds, metadata_list, args.total_spec_columns)

    # ===== Majority Voting =====
    print('starting majority voting')
    voted_preds = majority_voting_across_trials_efficient(final_preds, overlap_threshold=0.001)

    # Anzahl Calls nach Voting
    total = sum(len(preds["cluster"]) for preds in voted_preds.values())
    print(f"Total calls after voting: {total}")

    # ===== Save as .json =====
    with open(args.output_json, "w") as f:
        json.dump(voted_preds, f, indent=2)

    print(f"Inference complete. Results saved to {args.output_json}")