import argparse
import json
import os
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from whisperformer_dataset import WhisperFormerDataset
from whisperformer_model import WhisperFormer
#from wf_model_old import WhisperFormer
#from model_linear import WhisperFormer
from transformers import WhisperModel
from datautils import get_audio_and_label_paths_from_folders, load_data, get_cluster_codebook, FIXED_CLUSTER_CODEBOOK, ID_TO_CLUSTER
from datautils import slice_audios_and_labels
from whisperformer_train import collate_fn  # Reuse collate function from training
import numpy as np
from collections import defaultdict
import torch 
from transformers import WhisperFeatureExtractor


def load_trained_whisperformer(checkpoint_path, num_classes, device):
    """Load the WhisperFormer model with Whisper encoder and trained weights."""
    whisper_model = WhisperModel.from_pretrained("openai/whisper-small", local_files_only=True)
    encoder = whisper_model.encoder

    model = WhisperFormer(encoder, num_classes=num_classes)
    model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    model.to(device)
    model.eval()
    return model



def nms_1d_torch(intervals: torch.Tensor, iou_threshold):
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
    out = intervals[torch.tensor(keep, dtype=torch.long, device=intervals.device)]
    #out = intervals[keep]
    if out.ndim == 1:   # turn single interval into [1,3]
        out = out.unsqueeze(0)
    return out



def run_inference_new(model, dataloader, device, threshold, iou_threshold):
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

                B, T, C = class_preds.shape

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

def evaluate_detection_metrics_with_false_class(labels, predictions, overlap_tolerance=0.0001):
    """
    Berechnet TP, FP, FN, FC und F1 für gegebene Labels und Predictions.
    labels, predictions: Dicts mit keys 'onset', 'offset', 'cluster' (Listen)
    overlap_tolerance: Mindestüberlappung (0...1), damit ein Match zählt
    Rückgabe: dict mit keys 'tp', 'fp', 'fn', 'fc', 'f1', 'precision', 'recall'
    """
    label_onsets = np.array(labels['onset'])
    label_offsets = np.array(labels['offset'])
    label_clusters = np.array(labels['cluster'])

    pred_onsets = np.array(predictions['onset'])
    pred_offsets = np.array(predictions['offset'])
    pred_clusters = np.array(predictions['cluster'])

    matched_labels = set()
    matched_preds = set()
    false_class = 0

    # True Positives & False Class: Prediction und Label überlappen ausreichend
    for p_idx, (po, pf, pc) in enumerate(zip(pred_onsets, pred_offsets, pred_clusters)):
        for l_idx, (lo, lf, lc) in enumerate(zip(label_onsets, label_offsets, label_clusters)):
            if l_idx in matched_labels or p_idx in matched_preds:
                continue
            # Überlappung berechnen
            intersection = max(0, min(pf, lf) - max(po, lo))
            union = max(pf, lf) - min(po, lo)
            overlap_ratio = intersection / union if union > 0 else 0
            if overlap_ratio > overlap_tolerance:
                matched_labels.add(l_idx)
                matched_preds.add(p_idx)
                #if str(pc) == str(lc):
                #    matched_labels.add(l_idx)
                #    matched_preds.add(p_idx)
                #else:
                #    matched_labels.add(l_idx)
                #    matched_preds.add(p_idx)
                #    false_class += 1
                break

    tp = len(matched_labels) - false_class
    fp = len(pred_onsets) - len(matched_preds)
    fn = len(label_onsets) - len(matched_labels)
    fc = false_class
    precision = tp / (tp + fp + fc) if (tp + fp + fc) > 0 else 0
    recall    = tp / (tp + fn + fc) if (tp + fn + fc) > 0 else 0
    f1        = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

    return {
        'tp': tp,
        'fp': fp,
        'fn': fn,
        'fc': fc,
        'precision': precision,
        'recall': recall,
        'f1': f1
    }

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint_path", required=True, help="Path to the .pth trained model")
    parser.add_argument("--audio_folder", required=True)
    parser.add_argument("--label_folder", required=True)  
    parser.add_argument("--output_json", default="inference_results.json")
    parser.add_argument("--total_spec_columns", type=int, default=3000)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--num_classes", type=int, default=1)
    parser.add_argument("--threshold", type=float, default=0.35)
    parser.add_argument("--iou_threshold", type=float, default=0.1)
    parser.add_argument("--labels")
    parser.add_argument("--make_equal", nargs="+", default = None )


    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    # ===== Data loading =====
    audio_paths, label_paths = get_audio_and_label_paths_from_folders(args.audio_folder, args.label_folder)
    
    # Create a cluster codebook to use load_data
    #cluster_codebook = get_cluster_codebook(label_paths, initial_cluster_codebook={}, make_equal = args.make_equal)
    cluster_codebook = FIXED_CLUSTER_CODEBOOK
    id_to_cluster = ID_TO_CLUSTER


    # Mapping ID -> Name
    #id_to_cluster = {v: k for k, v in cluster_codebook.items()}
    #id_to_cluster = {}
    #for name, cid in cluster_codebook.items():
    #    if cid not in id_to_cluster:
    #        id_to_cluster[cid] = name  # nur erster Name pro ID
        
    # Load audio + labels
    audio_list, label_list = load_data(audio_paths, label_paths, cluster_codebook=cluster_codebook, n_threads=1)
    
    # Slice to fit model spec length
    audio_list, label_list, metadata_list = slice_audios_and_labels(audio_list, label_list, args.total_spec_columns)

    feature_extractor = WhisperFeatureExtractor.from_pretrained("openai/whisper-small", local_files_only=True)

    dataset = WhisperFormerDataset(audio_list, label_list, args.total_spec_columns, feature_extractor, args.num_classes)
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, 
                            collate_fn=collate_fn, drop_last=False)


    batch = next(iter(dataloader))
    x =batch['segments']
    # Indizes der nicht-null Elemente finden
    indices = torch.nonzero(x, as_tuple=False)

    # Ausgabe der Positionen und Werte
    for idx in indices:
        value = x[tuple(idx)]
        #print(f"Position: {tuple(idx.tolist())}, Wert: {value.item()}")

    # ===== Model loading =====
    model = load_trained_whisperformer(args.checkpoint_path, args.num_classes, args.device)

    # ===== Inference =====
    all_preds = run_inference_new(model, dataloader, args.device, args.threshold, args.iou_threshold)



    # Map predictions zurück zu den Original-Audios
    grouped_preds = defaultdict(list)

    for pred, meta in zip(all_preds, metadata_list):
        grouped_preds[meta["original_idx"]].append({
            "segment_idx": meta["segment_idx"],
            "class": pred["class"],
            "intervals": pred["intervals"]  
        })
        

    # ===== Reconstruction =====

    sec_per_col = 0.02
    # Wir erwarten nur eine Datei, also keine Schleife
    segs = list(grouped_preds.values())[0]
    segs_sorted = sorted(segs, key=lambda x: x["segment_idx"])

    classes, onsets, offsets, scores = [], [], [], []

    for seg in segs_sorted:
        offset_cols = seg["segment_idx"] * (args.total_spec_columns/2)
        for (start_col, end_col, score) in seg["intervals"]:
            start_sec = (offset_cols + start_col) * sec_per_col
            end_sec   = (offset_cols + end_col)   * sec_per_col

            class_id = seg["class"]
            class_name = id_to_cluster.get(class_id)

            classes.append(class_name)
            onsets.append(float(start_sec))
            offsets.append(float(end_sec))
            scores.append(float(score))

    # jetzt ein einfaches dict statt verschachtelt
    final_preds = {
        "onset": onsets,
        "offset": offsets,
        "cluster": classes,
        "score": scores
    }
    #total = sum(len(preds["cluster"]) for preds in final_preds.values())
    #print(f'Found {total} predictions.')


    # ===== Save as .json =====
    with open(args.output_json, "w") as f:
        json.dump(final_preds, f, indent=2)

    print(f"Inference complete. Results saved to {args.output_json}")

    with open(args.labels, 'r') as f:
        labels = json.load(f)

    metrics = evaluate_detection_metrics_with_false_class(labels, final_preds, overlap_tolerance=0.001)

    # Datei öffnen (z. B. im aktuellen Ordner)
    checkpoint_dir = os.path.dirname(args.checkpoint_path)
    output_file = os.path.join(checkpoint_dir, "metrics_results.txt")
    with open(output_file, "w") as f:
        f.write(f"True Positives: {metrics['tp']}\n")
        f.write(f"False Positives: {metrics['fp']}\n")
        f.write(f"False Negatives: {metrics['fn']}\n")
        f.write(f"False Class: {metrics['fc']}\n")
        f.write(f"Precision: {metrics['precision']:.4f}\n")
        f.write(f"Recall: {metrics['recall']:.4f}\n")
        f.write(f"F1-Score: {metrics['f1']:.4f}\n")

    print(f"Evaluation metrics wurden in '{output_file}' gespeichert.")


# Es fehlt: Überlappungen und Majority Voting!