import argparse
import json
import os
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from datetime import datetime
from collections import defaultdict
import numpy as np

from ..whisperformer.dataset import WhisperFormerDatasetQuality
from ..whisperformer.model import WhisperFormer
from transformers import WhisperModel, WhisperFeatureExtractor
from ..datautils import (
    get_audio_and_label_paths_from_folders,
    load_data,
    slice_audios_and_labels,
    FIXED_CLUSTER_CODEBOOK,
    ID_TO_CLUSTER
)
from ..whisperformer.train import collate_fn
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay


# ==================== MODEL LOADING ====================

def load_trained_whisperformer(checkpoint_path, num_classes, device):
    """Load a trained WhisperFormer from checkpoint.

    Args:
        checkpoint_path: Path to the state dict checkpoint.
        num_classes: Number of output classes for the head.
        device: Device to load the model to.

    Returns:
        WhisperFormer model in eval mode.
    """
    whisper_model = WhisperModel.from_pretrained("openai/whisper-small", local_files_only=True)
    encoder = whisper_model.encoder
    model = WhisperFormer(encoder, num_classes=num_classes)
    model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    model.to(device)
    model.eval()
    return model



# ==================== NMS ====================

def nms_1d_torch(intervals: torch.Tensor, iou_threshold):
    """Non-maximum suppression for 1D intervals (start, end, score). Keeps high-score non-overlapping intervals.

    Args:
        intervals: Tensor of shape (N, 3) with (start, end, score).
        iou_threshold: Intervals with IoU above this are suppressed.

    Returns:
        Tensor of kept intervals, shape (K, 3).
    """
    if intervals.numel() == 0:
        return intervals.new_zeros((0, 3))
    starts, ends, scores = intervals[:, 0], intervals[:, 1], intervals[:, 2]
    order = scores.argsort(descending=True)
    keep = []

    while order.numel() > 0:
        i = order[0]
        keep.append(i)
        if order.numel() == 1:
            break
        ss = torch.maximum(starts[i], starts[order[1:]])
        ee = torch.minimum(ends[i], ends[order[1:]])
        inter = torch.clamp(ee - ss, min=0)
        union = (ends[i] - starts[i]) + (ends[order[1:]] - starts[order[1:]]) - inter
        iou = inter / union
        inds = (iou <= iou_threshold).nonzero(as_tuple=False).squeeze(1)
        order = order[inds + 1]

    out = intervals[torch.tensor(keep, dtype=torch.long, device=intervals.device)]
    if out.ndim == 1:
        out = out.unsqueeze(0)
    return out


# ==================== INFERENCE ====================

def run_inference_new(model, dataloader, device, threshold, iou_threshold):
    all_preds = []
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Inference"):
            for k, v in batch.items():
                if isinstance(v, torch.Tensor):
                    batch[k] = v.to(device, non_blocking=True)

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
                                end = t + regr_preds[b, t, 1]
                                interval = torch.stack([start, end, score])
                                intervals.append(interval)

                        if intervals:
                            intervals = torch.stack(intervals)
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


# ==================== METRICS ====================


def evaluate_detection_metrics_with_false_class(labels, predictions, overlap_tolerance=0.001):
    """Compute TP, FP, FN, FC, precision, recall, F1 (basic version, no fp_scores/fn_qualities)."""
    label_onsets = np.array(labels['onset'])
    label_offsets = np.array(labels['offset'])
    label_clusters = np.array(labels['cluster'])

    pred_onsets = np.array(predictions['onset'])
    pred_offsets = np.array(predictions['offset'])
    pred_clusters = np.array(predictions['cluster'])

    matched_labels = set()
    matched_preds = set()
    false_class = 0

    for p_idx, (po, pf, pc) in enumerate(zip(pred_onsets, pred_offsets, pred_clusters)):
        for l_idx, (lo, lf, lc) in enumerate(zip(label_onsets, label_offsets, label_clusters)):
            if l_idx in matched_labels or p_idx in matched_preds:
                continue
            intersection = max(0, min(pf, lf) - max(po, lo))
            union = max(pf, lf) - min(po, lo)
            overlap_ratio = intersection / union if union > 0 else 0
            if overlap_ratio > overlap_tolerance:
                matched_labels.add(l_idx)
                matched_preds.add(p_idx)
                if str(pc) != str(lc):
                    false_class += 1
                break

    tp = len(matched_labels) - false_class
    fp = len(pred_onsets) - len(matched_preds)
    fn = len(label_onsets) - len(matched_labels)
    fc = false_class
    precision = tp / (tp + fp + fc) if (tp + fp + fc) > 0 else 0
    recall = tp / (tp + fn + fc) if (tp + fn + fc) > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

    return {'tp': tp, 'fp': fp, 'fn': fn, 'fc': fc,
            'precision': precision, 'recall': recall, 'f1': f1}

def evaluate_detection_metrics_with_false_class(labels, predictions, overlap_tolerance=0.001):
    label_onsets = np.array(labels['onset'])
    label_offsets = np.array(labels['offset'])
    label_clusters = np.array(labels['cluster'])
    label_qualities = np.array(labels.get('quality', ['unknown'] * len(label_onsets)))

    pred_onsets = np.array(predictions['onset'])
    pred_offsets = np.array(predictions['offset'])
    pred_clusters = np.array(predictions['cluster'])
    pred_scores = np.array(predictions.get('score', [0.0] * len(pred_onsets)))

    matched_labels = set()
    matched_preds = set()
    false_class = 0
    fp_scores = []           # Scores of all false positives
    fn_qualities = []        # Quality classes of all false negatives

    for p_idx, (po, pf, pc) in enumerate(zip(pred_onsets, pred_offsets, pred_clusters)):
        for l_idx, (lo, lf, lc) in enumerate(zip(label_onsets, label_offsets, label_clusters)):
            if l_idx in matched_labels or p_idx in matched_preds:
                continue
            intersection = max(0, min(pf, lf) - max(po, lo))
            union = max(pf, lf) - min(po, lo)
            overlap_ratio = intersection / union if union > 0 else 0
            if overlap_ratio > overlap_tolerance:
                matched_labels.add(l_idx)
                matched_preds.add(p_idx)
                if str(pc) != str(lc):
                    false_class += 1
                break

    # False positives = all unmatched predictions
    for p_idx in range(len(pred_onsets)):
        if p_idx not in matched_preds:
            fp_scores.append(pred_scores[p_idx])

    # False negatives = all unmatched labels
    for l_idx in range(len(label_onsets)):
        if l_idx not in matched_labels:
            fn_qualities.append(label_qualities[l_idx])

    tp = len(matched_labels) - false_class
    fp = len(pred_onsets) - len(matched_preds)
    fn = len(label_onsets) - len(matched_labels)
    fc = false_class

    precision = tp / (tp + fp + fc) if (tp + fp + fc) > 0 else 0
    recall = tp / (tp + fn + fc) if (tp + fn + fc) > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

    return {
        'tp': tp,
        'fp': fp,
        'fn': fn,
        'fc': fc,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'fp_scores': fp_scores,
        'fn_qualities': fn_qualities
    }

def evaluate_detection_metrics_with_false_class_qualities(labels, predictions, overlap_tolerance=0.001, allowed_qualities={1, 2}):
    """Compute detection metrics (TP, FP, FN, FC, precision, recall, F1) with optional quality filtering.

    If allowed_qualities is set, ground-truth labels are filtered to those quality values before matching.

    Args:
        labels: Dict with 'onset', 'offset', 'cluster', 'quality'.
        predictions: Dict with 'onset', 'offset', 'cluster'.
        overlap_tolerance: Minimum overlap ratio for a match.
        allowed_qualities: If not None, restrict GT to these quality values.

    Returns:
        dict: gtp, pp, tp, fp, fn, fc, precision, recall, f1.
    """
    # If quality filter is set: filter GT
    label_onsets   = labels['onset']
    label_offsets  = labels['offset']
    label_clusters = labels['cluster']
    label_qualities = labels['quality']
    #print(label_qualities)

    if allowed_qualities is not None:
        # Normalize to strings so that {1,2} or {"1","2"} both work
        allowed_str = set(str(q) for q in allowed_qualities)
        qual_str = np.array([str(q) for q in label_qualities])
        mask = np.array([q in allowed_str for q in qual_str], dtype=bool)


        label_onsets = np.array(label_onsets)[mask]
        label_offsets = np.array(label_offsets)[mask]
        label_clusters = np.array(label_clusters)[mask]
        # label_qualities not needed after this

    # Load predictions
    pred_onsets = np.array(predictions['onset'])
    pred_offsets = np.array(predictions['offset'])
    pred_clusters = np.array(predictions['cluster'])

    
    matched_labels = set()
    matched_preds = set()
    false_class = 0

    gtp = len(label_onsets)      # Now only quality in allowed set
    pp  = len(pred_onsets)

    # Matching
    for p_idx, (po, pf, pc) in enumerate(zip(pred_onsets, pred_offsets, pred_clusters)):
        for l_idx, (lo, lf, lc) in enumerate(zip(label_onsets, label_offsets, label_clusters)):
            if l_idx in matched_labels or p_idx in matched_preds:
                continue
            inter = max(0.0, min(pf, lf) - max(po, lo))
            union = max(pf, lf) - min(po, lo)
            ov = inter / union if union > 0 else 0.0
            if ov > overlap_tolerance:
                matched_labels.add(l_idx)
                matched_preds.add(p_idx)
                if str(pc) != str(lc):
                    false_class += 1
                break

    tp = len(matched_labels) - false_class
    fp = len(pred_onsets) - len(matched_preds)
    fn = len(label_onsets) - len(matched_labels)
    fc = false_class

    precision = tp / (tp + fp + fc) if (tp + fp + fc) > 0 else 0.0
    recall    = tp / (tp + fn + fc) if (tp + fn + fc) > 0 else 0.0
    f1        = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

    return {
        'gtp': gtp, 'pp': pp,
        'tp': tp, 'fp': fp, 'fn': fn, 'fc': fc,
        'precision': precision, 'recall': recall, 'f1': f1
    }


# ==================== MAIN ====================

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint_path", required=True)
    parser.add_argument("--audio_folder", required=True)
    parser.add_argument("--label_folder", required=True)
    parser.add_argument("--output_dir", default="inference_outputs")
    parser.add_argument("--total_spec_columns", type=int, default=3000)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--num_classes", type=int, default=1)
    parser.add_argument("--threshold", type=float, default=0.35)
    parser.add_argument("--iou_threshold", type=float, default=0.1)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    # Create timestamped subfolder
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    save_dir = os.path.join(args.output_dir, timestamp)
    os.makedirs(save_dir, exist_ok=True)

    # Save arguments
    args_path = os.path.join(save_dir, "run_arguments.json")
    with open(args_path, "w") as f:
        json.dump(vars(args), f, indent=2)
    print(f"Arguments saved to: {args_path}")

    #os.makedirs(args.output_dir, exist_ok=True)

    audio_paths, label_paths = get_audio_and_label_paths_from_folders(args.audio_folder, args.label_folder)
    cluster_codebook = FIXED_CLUSTER_CODEBOOK
    id_to_cluster = ID_TO_CLUSTER

    model = load_trained_whisperformer(args.checkpoint_path, args.num_classes, args.device)
    feature_extractor = WhisperFeatureExtractor.from_pretrained("openai/whisper-small", local_files_only=True)

    all_labels = {"onset": [], "offset": [], "cluster": [], "quality": []}
    all_preds_final  = {"onset": [], "offset": [], "cluster": [], "score": []}

    for audio_path, label_path in zip(audio_paths, label_paths):
        print(f"\n===== Processing {os.path.basename(audio_path)} =====")
        audio_list, label_list = load_data([audio_path], [label_path], cluster_codebook=cluster_codebook, n_threads=1)
        audio_list, label_list, metadata_list = slice_audios_and_labels(audio_list, label_list, args.total_spec_columns)

        dataset = WhisperFormerDatasetQuality(audio_list, label_list, args.total_spec_columns, feature_extractor, args.num_classes)
        dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False,
                                collate_fn=collate_fn, drop_last=False)

        all_preds = run_inference_new(model, dataloader, args.device, args.threshold, args.iou_threshold)

        grouped_preds = defaultdict(list)
        for pred, meta in zip(all_preds, metadata_list):
            grouped_preds[meta["original_idx"]].append({
                "segment_idx": meta["segment_idx"],
                "class": pred["class"],
                "intervals": pred["intervals"]
            })

        # Reconstruction: map segment indices and columns back to seconds
        sec_per_col = 0.02
        segs_sorted = sorted(grouped_preds[0], key=lambda x: x["segment_idx"])
        classes, onsets, offsets, scores = [], [], [], []

        for seg in segs_sorted:
            offset_cols = seg["segment_idx"] * (args.total_spec_columns / 2)
            for (start_col, end_col, score) in seg["intervals"]:
                start_sec = (offset_cols + start_col) * sec_per_col
                end_sec = (offset_cols + end_col) * sec_per_col
                classes.append(id_to_cluster.get(seg["class"]))
                onsets.append(float(start_sec))
                offsets.append(float(end_sec))
                scores.append(float(score))

        final_preds = {"onset": onsets, "offset": offsets, "cluster": classes, "score": scores}

        # Save predictions per file
        base_name = os.path.splitext(os.path.basename(audio_path))[0]
        json_path = os.path.join(save_dir, f"{base_name}_preds.json")
        with open(json_path, "w") as f:
            json.dump(final_preds, f, indent=2)
        print(f"Predictions saved to {json_path}")

        # Load labels
        with open(label_path, "r") as f:
            labels = json.load(f)
        
        clusters = labels["cluster"]
        labels["cluster"] = [ID_TO_CLUSTER[FIXED_CLUSTER_CODEBOOK[c]] for c in clusters]

        # Add quality classes
        if "quality" in labels:
            quality_list = labels["quality"]
        else:
            quality_list = ["unknown"] * len(labels["onset"])


        # Fill global collectors
        all_labels["onset"].extend(labels["onset"])
        all_labels["offset"].extend(labels["offset"])
        all_labels["cluster"].extend(labels["cluster"])
        all_labels["quality"].extend(quality_list)

        all_preds_final["onset"].extend(final_preds["onset"])
        all_preds_final["offset"].extend(final_preds["offset"])
        all_preds_final["cluster"].extend(final_preds["cluster"])
        all_preds_final["score"].extend(final_preds["score"])

    metrics = evaluate_detection_metrics_with_false_class_qualities(all_labels, all_preds_final)
    """
    # compute racll for each quality class
    quality_classes = sorted(set(all_labels["quality"]))
    recall_per_quality = {}

    for q in quality_classes:
        total_q = sum(1 for qual in all_labels["quality"] if qual == q)
        missed_q = sum(1 for qual in metrics['fn_qualities'] if qual == q)
        recall_q = (total_q - missed_q) / total_q if total_q > 0 else 0.0
        recall_per_quality[q] = recall_q

    from collections import Counter

    # Frequencies of FN quality classes
    fn_quality_counts = Counter(metrics['fn_qualities'])
    """
    metrics_path = os.path.join(save_dir, "metrics_all_qualities.txt")

    with open(metrics_path, "w") as f:
        f.write(f"Global metrics for threshold {args.threshold} and iou threshold {args.iou_threshold}:\n")
        f.write(f"TP: {metrics['tp']}\n")
        f.write(f"FP: {metrics['fp']}\n")
        f.write(f"FN: {metrics['fn']}\n")
        f.write(f"FC: {metrics['fc']}\n")
        f.write(f"Precision: {metrics['precision']:.4f}\n")
        f.write(f"Recall:    {metrics['recall']:.4f}\n")
        f.write(f"F1-Score:  {metrics['f1']:.4f}\n\n")
        """
        f.write("Recall per quality class:\n")
        for q, r in recall_per_quality.items():
            f.write(f"  {q}: {r:.4f}\n")

        f.write("\nScores of false positives:\n")
        f.write(", ".join([f"{s:.3f}" for s in metrics['fp_scores']]) + "\n")

        f.write("\nQuality-Klassen der False Negatives (Häufigkeit):\n")
        for q, count in fn_quality_counts.items():
            f.write(f"  Class {q}: {count}\n")
    """
    print(f"Global metrics saved to {metrics_path}")

    from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

    # Confusion matrix for classes + None
    all_classes = sorted(set(all_labels["cluster"]))   # All true classes
    class_names = all_classes + ["None"]               # Additional None class

    y_true = []
    y_pred = []

    matched_labels = set()
    matched_preds = set()

    # Match events as in evaluate_detection_metrics_with_false_class
    overlap_tolerance = 0.001
    for p_idx, (po, pf, pc) in enumerate(zip(all_preds_final['onset'],
                                            all_preds_final['offset'],
                                            all_preds_final['cluster'])):
        for l_idx, (lo, lf, lc) in enumerate(zip(all_labels['onset'],
                                                all_labels['offset'],
                                                all_labels['cluster'])):
            if l_idx in matched_labels or p_idx in matched_preds:
                continue
            intersection = max(0, min(pf, lf) - max(po, lo))
            union = max(pf, lf) - min(po, lo)
            overlap_ratio = intersection / union if union > 0 else 0
            if overlap_ratio > overlap_tolerance:
                matched_labels.add(l_idx)
                matched_preds.add(p_idx)
                # Correctly localized: check class
                if lc == pc:
                    y_true.append(lc)
                    y_pred.append(pc)
                else:
                    # Wrong class
                    y_true.append(lc)
                    y_pred.append(pc)

    # False negatives (labels without match)
    for l_idx, lc in enumerate(all_labels['cluster']):
        if l_idx not in matched_labels:
            y_true.append(lc)
            y_pred.append("None")

    # False positives (predictions without match)
    for p_idx, pc in enumerate(all_preds_final['cluster']):
        if p_idx not in matched_preds:
            y_true.append("None")
            y_pred.append(pc)

    # Compute and plot confusion matrix
    cm = confusion_matrix(y_true, y_pred, labels=class_names)

    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_names)
    fig, ax = plt.subplots(figsize=(8, 6))
    disp.plot(ax=ax, cmap="Blues", xticks_rotation=45)
    plt.title("Confusion matrix (classes + None)")
    plt.tight_layout()

    cm_path = os.path.join(save_dir, "confusion_matrix_classes_none.png")
    plt.savefig(cm_path, dpi=150)
    plt.close()

    print(f"Confusion matrix saved to {cm_path}")
