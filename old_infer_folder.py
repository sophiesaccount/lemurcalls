import argparse
import json
import os
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from datetime import datetime
from collections import defaultdict
import numpy as np

from whisperformer_dataset import WhisperFormerDataset
from whisperformer_model import WhisperFormer
from transformers import WhisperModel, WhisperFeatureExtractor
from datautils import (
    get_audio_and_label_paths_from_folders,
    load_data,
    slice_audios_and_labels,
    FIXED_CLUSTER_CODEBOOK,
    ID_TO_CLUSTER
)
from whisperformer_train import collate_fn
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from sklearn.metrics import precision_score, recall_score, f1_score


# ==================== MODEL LOADING ====================

def load_trained_whisperformer(checkpoint_path, num_classes, device):
    whisper_model = WhisperModel.from_pretrained("openai/whisper-small", local_files_only=True)
    encoder = whisper_model.encoder
    model = WhisperFormer(encoder, num_classes=num_classes)
    model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    model.to(device)
    model.eval()
    return model

# ==================== CONFUSION MATRIX ====================
def get_confusion_inputs(labels, predictions, overlap_tolerance=0.001):
    """
    Gibt zwei Listen zurück: true_labels, pred_labels für Confusion Matrix.
    - Labels ohne Match -> pred_label = "none"  (False Negatives)
    - Predictions ohne Match -> true_label = "none", pred_label = erkannte Klasse (False Positives)
    """

    true_labels = []
    pred_labels = []

    matched_preds = set()

    # 1) Labels mit Predictions abgleichen
    for l_idx, (l_on, l_off, l_cl) in enumerate(zip(labels['onset'], labels['offset'], labels['cluster'])):
        overlaps = []
        for p_idx, (p_on, p_off, p_cl) in enumerate(zip(predictions['onset'], predictions['offset'], predictions['cluster'])):
            intersection = max(0, min(l_off, p_off) - max(l_on, p_on))
            union = max(l_off, p_off) - min(l_on, p_on)
            overlap_ratio = intersection / union if union > 0 else 0

            if overlap_ratio > overlap_tolerance:
                overlaps.append((p_idx, p_cl))

        if overlaps:
            best_idx, best_class = overlaps[0]
            matched_preds.add(best_idx)
            true_labels.append(l_cl)
            pred_labels.append(best_class)
        else:
            # kein Match → False Negative
            true_labels.append(l_cl)
            pred_labels.append("none")

    # 2) Alle Predictions ohne Match → False Positives
    for p_idx, p_cl in enumerate(predictions['cluster']):
        if p_idx not in matched_preds:
            true_labels.append("none")
            pred_labels.append(p_cl)

    return true_labels, pred_labels

# ==================== NMS ====================

def nms_1d_torch(intervals: torch.Tensor, iou_threshold):
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

    os.makedirs(args.output_dir, exist_ok=True)

    audio_paths, label_paths = get_audio_and_label_paths_from_folders(args.audio_folder, args.label_folder)
    cluster_codebook = FIXED_CLUSTER_CODEBOOK
    id_to_cluster = ID_TO_CLUSTER

    model = load_trained_whisperformer(args.checkpoint_path, args.num_classes, args.device)
    feature_extractor = WhisperFeatureExtractor.from_pretrained("openai/whisper-small", local_files_only=True)

    # Für globale Metriken nach Quality
    quality_classes = ["1", "2", "3"]
    all_true_labels_by_quality = {q: [] for q in quality_classes}
    all_pred_labels_by_quality = {q: [] for q in quality_classes}

    for audio_path, label_path in zip(audio_paths, label_paths):
        print(f"\n===== Processing {os.path.basename(audio_path)} =====")
        audio_list, label_list = load_data([audio_path], [label_path], cluster_codebook=cluster_codebook, n_threads=1)
        audio_list, label_list, metadata_list = slice_audios_and_labels(audio_list, label_list, args.total_spec_columns)

        dataset = WhisperFormerDataset(audio_list, label_list, args.total_spec_columns, feature_extractor, args.num_classes)
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

        # Rekonstruktion
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

        # Predictions pro Datei speichern
        base_name = os.path.splitext(os.path.basename(audio_path))[0]
        json_path = os.path.join(args.output_dir, f"{base_name}_preds.json")
        with open(json_path, "w") as f:
            json.dump(final_preds, f, indent=2)
        print(f"✅ Predictions saved to {json_path}")

        # Labels laden
        with open(label_path, "r") as f:
            labels = json.load(f)

        # Labels nach Quality sammeln
        if "quality" in labels:
            for q in quality_classes:
                indices = [i for i, val in enumerate(labels["quality"]) if val == q]
                if not indices:
                    continue
                filtered_labels = {
                    "onset": [labels["onset"][i] for i in indices],
                    "offset": [labels["offset"][i] for i in indices],
                    "cluster": [ID_TO_CLUSTER[FIXED_CLUSTER_CODEBOOK[labels["cluster"][i]]] for i in indices],
                }

                t_labels, p_labels = get_confusion_inputs(filtered_labels, final_preds)
                all_true_labels_by_quality[q].extend(t_labels)
                all_pred_labels_by_quality[q].extend(p_labels)

    # ==================== Globale Metriken & Confusion pro Quality ====================
    for q in quality_classes:
        true_labels = all_true_labels_by_quality[q]
        pred_labels = all_pred_labels_by_quality[q]
        if not true_labels:
            continue  # keine Daten für diese Quality-Klasse

        # Alle Klassen + "none" für Predictions ohne Match
        all_classes = sorted(set(true_labels) | set(pred_labels) | {"none"})

        # Confusion-Matrix mit "none" behandeln
        cm = confusion_matrix(true_labels, pred_labels, labels=all_classes)
        disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=all_classes)

        # Confusion-Matrix speichern
        plt.figure(figsize=(10, 10))
        disp.plot(cmap=plt.cm.Blues, xticks_rotation=45)
        plt.title(f"Confusion Matrix - Quality {q}")
        confusion_path = os.path.join(args.output_dir, f"confusion_matrix_quality_{q}.png")
        plt.savefig(confusion_path, bbox_inches='tight', dpi=300)
        plt.close()
        print(f"✅ Confusion Matrix for Quality {q} saved to {confusion_path}")

        # -----> Pro Klasse Metriken berechnen
        metrics_path = os.path.join(args.output_dir, f"metrics_quality_{q}.txt")
        with open(metrics_path, "w") as f:
            f.write("Class-wise metrics:\n")
            for i, cls in enumerate(all_classes):
                tp = cm[i, i]
                fp = cm[:, i].sum() - tp
                fn = cm[i, :].sum() - tp
                tn = cm.sum() - (tp + fp + fn)

                precision = tp / (tp + fp) if (tp + fp) > 0 else 0
                recall = tp / (tp + fn) if (tp + fn) > 0 else 0
                f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

                f.write(f"\nClass: {cls}\n")
                f.write(f"  TP: {tp}\n")
                f.write(f"  FP: {fp}\n")
                f.write(f"  FN: {fn}\n")
                f.write(f"  TN: {tn}\n")
                f.write(f"  Precision: {precision:.4f}\n")
                f.write(f"  Recall:    {recall:.4f}\n")
                f.write(f"  F1-Score:  {f1:.4f}\n")

            # -----> Optional: Makro- und Mikro-Durchschnitt
            macro_f1 = f1_score(true_labels, pred_labels, labels=all_classes, average='macro')
            weighted_f1 = f1_score(true_labels, pred_labels, labels=all_classes, average='weighted')

            f.write("\nOverall metrics:\n")
            f.write(f"Macro-F1:    {macro_f1:.4f}\n")
            f.write(f"Weighted-F1: {weighted_f1:.4f}\n")

        print(f"✅ Metrics for Quality {q} saved to {metrics_path}")
    
    # Alle Quality-Listen zu einer globalen Liste zusammenführen
    all_true_labels_global = []
    all_pred_labels_global = []

    for q in quality_classes:
        all_true_labels_global.extend(all_true_labels_by_quality[q])
        all_pred_labels_global.extend(all_pred_labels_by_quality[q])
    
    from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, f1_score

    # alle Klassen + "none"
    all_classes = sorted(set(all_true_labels_global) |
                        set(all_pred_labels_global) | {"none"})

    cm_global = confusion_matrix(all_true_labels_global,
                                all_pred_labels_global,
                                labels=all_classes)

    # Plot und speichern
    plt.figure(figsize=(10,10))
    disp = ConfusionMatrixDisplay(confusion_matrix=cm_global,
                                display_labels=all_classes)
    disp.plot(cmap=plt.cm.Blues, xticks_rotation=45)
    plt.title("Confusion Matrix – alle Quality-Klassen")
    plt.savefig(os.path.join(args.output_dir,
                            "confusion_matrix_all_qualities.png"),
                bbox_inches="tight", dpi=300)
    plt.close()

    metrics_path = os.path.join(args.output_dir, "metrics_all_qualities.txt")

    with open(metrics_path, "w") as f:
        f.write("Globale Class-wise Metriken:\n")
        for i, cls in enumerate(all_classes):
            tp = cm_global[i, i]
            fp = cm_global[:, i].sum() - tp
            fn = cm_global[i, :].sum() - tp
            tn = cm_global.sum() - (tp + fp + fn)

            precision = tp / (tp + fp) if (tp + fp) > 0 else 0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0
            f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

            f.write(f"\nClass: {cls}\n")
            f.write(f"  TP: {tp}\n")
            f.write(f"  FP: {fp}\n")
            f.write(f"  FN: {fn}\n")
            f.write(f"  TN: {tn}\n")
            f.write(f"  Precision: {precision:.4f}\n")
            f.write(f"  Recall:    {recall:.4f}\n")
            f.write(f"  F1-Score:  {f1:.4f}\n")

        # Makro- und Weighted-F1
        macro_f1 = f1_score(all_true_labels_global, all_pred_labels_global,
                            labels=all_classes, average="macro")
        weighted_f1 = f1_score(all_true_labels_global, all_pred_labels_global,
                            labels=all_classes, average="weighted")

        f.write("\nOverall metrics:\n")
        f.write(f"Macro-F1:    {macro_f1:.4f}\n")
        f.write(f"Weighted-F1: {weighted_f1:.4f}\n")

    print(f"✅ Globale Confusion-Matrix und Metriken gespeichert unter {metrics_path}")
