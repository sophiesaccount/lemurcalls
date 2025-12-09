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



# ==================== MODEL LOADING ====================

def load_trained_whisperformer(checkpoint_path, num_classes, device):
    whisper_model = WhisperModel.from_pretrained("openai/whisper-small", local_files_only=True)
    encoder = whisper_model.encoder
    model = WhisperFormer(encoder, num_classes=num_classes)
    model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    model.to(device)
    model.eval()
    return model


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

def temporal_iou(seg1, seg2):
    """
    Berechnet IoU für 1D-Segmente.
    seg1, seg2: [start, end]
    """
    inter_start = max(seg1[0], seg2[0])
    inter_end = min(seg1[1], seg2[1])
    inter = max(0.0, inter_end - inter_start)
    union = (seg1[1] - seg1[0]) + (seg2[1] - seg2[0]) - inter
    return inter / union if union > 0 else 0.0

def f1_score_1d(gt_segments, pred_segments, iou_threshold=0.5):
    """
    Berechnet F1-Score für 1D Segmente.
    
    gt_segments: Liste von [start, end] Ground-Truth-Segmenten
    pred_segments: Liste von [start, end] Vorhersagen
    iou_threshold: IoU-Schwelle für TP
    """
    gt_segments = [np.array(seg) for seg in gt_segments]
    pred_segments = [np.array(seg) for seg in pred_segments]

    matched_gt = set()
    tp = 0
    fp = 0

    # Sortiere Predictions optional nach Länge oder Confidence (falls Scores vorhanden)
    # hier gleich Reihenfolge, sonst max-score zuerst
    for pred in pred_segments:
        best_iou = 0
        best_gt_idx = None
        for i, gt in enumerate(gt_segments):
            if i in matched_gt:
                continue
            iou = temporal_iou(pred, gt)
            if iou > best_iou:
                best_iou = iou
                best_gt_idx = i
        if best_iou >= iou_threshold:
            tp += 1
            matched_gt.add(best_gt_idx)
        else:
            fp += 1

    fn = len(gt_segments) - len(matched_gt)

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

    return {"precision": precision, "recall": recall, "f1": f1, "tp": tp, "fp": fp, "fn": fn}


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

    # Output-Verzeichnis
    os.makedirs(args.output_dir, exist_ok=True)

    # Alle Audio + Label Dateien holen
    audio_paths, label_paths = get_audio_and_label_paths_from_folders(args.audio_folder, args.label_folder)

    # Feste Cluster-Mappings
    cluster_codebook = FIXED_CLUSTER_CODEBOOK
    id_to_cluster = ID_TO_CLUSTER

    # Modell laden
    model = load_trained_whisperformer(args.checkpoint_path, args.num_classes, args.device)
    feature_extractor = WhisperFeatureExtractor.from_pretrained("openai/whisper-small", local_files_only=True)

    # Durch alle Dateien loopen
    for audio_path, label_path in zip(audio_paths, label_paths):
        print(f"\n===== Processing {os.path.basename(audio_path)} =====")

        # Daten laden
        audio_list, label_list = load_data([audio_path], [label_path], cluster_codebook=cluster_codebook, n_threads=1)
        audio_list, label_list, metadata_list = slice_audios_and_labels(audio_list, label_list, args.total_spec_columns)

        dataset = WhisperFormerDataset(audio_list, label_list, args.total_spec_columns, feature_extractor, args.num_classes)
        dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False,
                                collate_fn=collate_fn, drop_last=False)

        # Inferenz
        all_preds = run_inference_new(model, dataloader, args.device, args.threshold, args.iou_threshold)

        # Gruppieren nach Original-Segmenten
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

        # Outputs speichern
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        base_name = os.path.splitext(os.path.basename(audio_path))[0]

        json_path = os.path.join(args.output_dir, f"{base_name}_preds_{timestamp}.json")
        with open(json_path, "w") as f:
            json.dump(final_preds, f, indent=2)
        print(f"✅ Predictions saved to {json_path}")



        # Labels laden
        with open(label_path, "r") as f:
            labels = json.load(f)


        quality_classes = ["1", "2", "3"]              # Quality-Level
        cluster_classes = sorted(set(labels["cluster"]))  # Alle vorkommenden Klassen

        metrics_by_quality_and_class = {}

        # --- Schleife über Quality ---
        if "quality" in labels:
            for q in quality_classes:
                indices_q = [i for i, val in enumerate(labels["quality"]) if val == q]
                if not indices_q:
                    continue

                # Labels, die zu dieser Quality gehören
                filtered_labels_q = {
                    "onset":  [labels["onset"][i]   for i in indices_q],
                    "offset": [labels["offset"][i]  for i in indices_q],
                    "cluster":[labels["cluster"][i] for i in indices_q],
                }

                metrics_by_quality_and_class[q] = {}

                # --- Schleife über Cluster ---
                for cls in cluster_classes:
                    # Alle GT-Segmente dieser Klasse + Quality
                    indices_c = [i for i, val in enumerate(filtered_labels_q["cluster"]) if val == cls]
                    if not indices_c:
                        continue

                    gt_cls = {
                        "onset":  [filtered_labels_q["onset"][i]  for i in indices_c],
                        "offset": [filtered_labels_q["offset"][i] for i in indices_c],
                        "cluster":[filtered_labels_q["cluster"][i]for i in indices_c],
                    }

                    # Auch die Predictions filtern (falls final_preds ein dict mit Klassen ist)
                    preds_cls = [p for p in final_preds if p["cluster"] == cls]

                    metrics = f1_score_1d(gt_cls, preds_cls, overlap_tolerance=0.5)
                    metrics_by_quality_and_class[q][cls] = metrics

        # Beispielausgabe
        for q, cls_dict in metrics_by_quality_and_class.items():
            print(f"\nQuality {q}:")
            for cls, m in cls_dict.items():
                print(f"  Class {cls}: F1={m['f1']:.3f}  P={m['precision']:.3f}  R={m['recall']:.3f}")
