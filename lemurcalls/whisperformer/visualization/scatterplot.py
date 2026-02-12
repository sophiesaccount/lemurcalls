import argparse
import json
import os
import torch
from torch.utils.data import DataLoader
from datetime import datetime
from collections import defaultdict
import numpy as np

from ..dataset import WhisperFormerDatasetQuality
from ..model import WhisperFormer
from transformers import WhisperModel, WhisperFeatureExtractor
from ...datautils import (
    get_audio_and_label_paths_from_folders,
    load_data,
    slice_audios_and_labels,
    FIXED_CLUSTER_CODEBOOK,
    ID_TO_CLUSTER
)
from ..train import collate_fn, nms_1d_torch, evaluate_detection_metrics_with_false_class_qualities, group_by_file
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import contextlib

def evaluate_detection_metrics_with_false_class_qualities(labels, predictions, overlap_tolerance, allowed_qualities = [1,2]):
    # Falls Qualitätsfilter gesetzt: GT filtern

    label_onsets   = labels['onset']
    label_offsets  = labels['offset']
    label_clusters = labels['cluster']
    label_qualities = labels['quality']
    
    #print(label_qualities)

    if allowed_qualities is not None:
        # alles zu int konvertieren (wenn möglich)
        try:
            allowed_ints = set(int(q) for q in allowed_qualities)
            label_ints = np.array([int(float(q)) for q in label_qualities])
            mask = np.array([q in allowed_ints for q in label_ints], dtype=bool)
        except ValueError:
            # fallback auf stringvergleich, falls nicht-numerische Qualitäten vorkommen
            allowed_str = set(str(q) for q in allowed_qualities)
            qual_str = np.array([str(q) for q in label_qualities])
            mask = np.array([q in allowed_str for q in qual_str], dtype=bool)

        label_onsets   = np.array(label_onsets)[mask]
        label_offsets  = np.array(label_offsets)[mask]
        label_clusters = np.array(label_clusters)[mask]


    # Predictions laden
    pred_onsets = np.array(predictions['onset'])
    pred_offsets = np.array(predictions['offset'])
    pred_clusters = np.array(predictions['cluster'])

    
    matched_labels = set()
    matched_preds = set()
    false_class = 0

    gtp = len(label_onsets)      # jetzt nur noch Quality∈allowed
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
                matched_labels
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


# ==================== MODEL LOADING ====================

def load_trained_whisperformer(checkpoint_path, num_classes, num_decoder_layers, num_head_layers, device):
    whisper_model = WhisperModel.from_pretrained("openai/whisper-small", local_files_only=True)
    encoder = whisper_model.encoder
    model = WhisperFormer(encoder, num_classes=num_classes, num_decoder_layers=num_decoder_layers, num_head_layers=num_head_layers )
    model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    model.to(device)
    model.eval()
    return model


# ==================== INFERENCE ====================

def run_inference_new(model, dataloader, device, threshold, iou_threshold, metadata_list):
    """
    Führt Inferenz durch und ordnet jede Vorhersage exakt dem Slice in metadata_list zu.
    Gibt eine Liste von Einträgen zurück:
    {
      "original_idx": int,
      "segment_idx": int,
      "preds": [ { "class": c, "intervals": [[start_col, end_col, score], ...] }, ... ]
    }
    """
    preds_by_slice = []
    slice_idx = 0
    model.eval()

    with torch.no_grad():
        for batch in dataloader:
            # Tensoren auf Device bringen
            for k, v in batch.items():
                if isinstance(v, torch.Tensor):
                    batch[k] = v.to(device, non_blocking=True)

            # Autocast nur auf CUDA aktivieren
            use_autocast = (isinstance(device, str) and device.startswith("cuda")) or (hasattr(device, "type") and device.type == "cuda")
            autocast_ctx = torch.amp.autocast(device_type="cuda", dtype=torch.float16) if use_autocast else contextlib.nullcontext()

            with autocast_ctx:
                class_preds, regr_preds = model(batch["input_features"])
                class_probs = torch.sigmoid(class_preds)

            B, T, C = class_preds.shape
            for b in range(B):
                # passendes Slice aus metadata_list holen
                meta = metadata_list[slice_idx]
                slice_idx += 1

                preds_per_class = []
                for c in range(C):
                    intervals = []
                    for t in range(T):
                        score = class_probs[b, t, c]
                        if float(score) > threshold:
                            start = t - regr_preds[b, t, 0]
                            end   = t + regr_preds[b, t, 1]
                            intervals.append(torch.stack([start, end, score]))

                    if len(intervals) > 0:
                        intervals = torch.stack(intervals)
                        intervals = nms_1d_torch(intervals, iou_threshold=iou_threshold)
                        intervals = intervals.cpu().tolist()
                    else:
                        intervals = []

                    preds_per_class.append({"class": c, "intervals": intervals})

                preds_by_slice.append({
                    "original_idx": meta["original_idx"],
                    "segment_idx": meta["segment_idx"],
                    "preds": preds_per_class
                })

    # Sanity-Check: Anzahl Slices sollte übereinstimmen
    assert len(preds_by_slice) == len(metadata_list), (
        f"Vorhersage-Liste ({len(preds_by_slice)}) ungleich Metadata-Liste ({len(metadata_list)}). "
        "Prüfen Sie, ob DataLoader shuffle=False ist und die Reihenfolge konsistent ist."
    )

    return preds_by_slice


def reconstruct_predictions(preds_by_slice, total_spec_columns, ID_TO_CLUSTER):
    """
    Rekonstruiert alle Vorhersagen aus Slice-Koordinaten in Datei-Zeitkoordinaten.
    Gibt ein Dict mit Listen zurück: {"onset": [], "offset": [], "cluster": [], "score": []}
    """
    grouped_preds = defaultdict(list)
    for ps in preds_by_slice:
        grouped_preds[ps["original_idx"]].append(ps)

    sec_per_col = 0.02
    cols_per_segment = total_spec_columns // 2  # T entspricht total_spec_columns/2

    all_preds_final = {"onset": [], "offset": [], "cluster": [], "score": []}

    # Über alle Originaldateien iterieren
    for orig_idx in sorted(grouped_preds.keys()):
        segs_sorted = sorted(grouped_preds[orig_idx], key=lambda x: x["segment_idx"])
        for seg in segs_sorted:
            offset_cols = seg["segment_idx"] * cols_per_segment
            for p in seg["preds"]:
                c = p["class"]
                for (start_col, end_col, score) in p["intervals"]:
                    start_sec = (offset_cols + start_col) * sec_per_col
                    end_sec   = (offset_cols + end_col)   * sec_per_col
                    all_preds_final["onset"].append(float(start_sec))
                    all_preds_final["offset"].append(float(end_sec))
                    # Map Klasse-ID -> Cluster-Label
                    #all_preds_final["cluster"].append(ID_TO_CLUSTER[c] if c in range(len(ID_TO_CLUSTER)) else "unknown")
                    all_preds_final["cluster"].append(ID_TO_CLUSTER.get(c, "unknown"))
                    all_preds_final["score"].append(float(score))

    return all_preds_final


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
    parser.add_argument("--overlap_tolerance", type=float, default=0.1)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--num_decoder_layers", type = int, default = 3)
    parser.add_argument("--num_head_layers", type = int, default = 2)
    parser.add_argument("--low_quality_value", type = float, default = 0.5)
    parser.add_argument("--allowed_qualities", default = [1,2])
    parser.add_argument("--num_workers", type = int, default = 1 )
    args = parser.parse_args()

    # === Zeitgestempelten Unterordner erstellen ===
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    save_dir = os.path.join(args.output_dir, timestamp)
    os.makedirs(save_dir, exist_ok=True)

    # === Argumente speichern ===
    args_path = os.path.join(save_dir, "run_arguments.json")
    with open(args_path, "w") as f:
        json.dump(vars(args), f, indent=2)
    print(f"✅ Argumente gespeichert unter: {args_path}")

    #os.makedirs(args.output_dir, exist_ok=True)

    audio_paths, label_paths = get_audio_and_label_paths_from_folders(args.audio_folder, args.label_folder)
    cluster_codebook = FIXED_CLUSTER_CODEBOOK
    id_to_cluster = ID_TO_CLUSTER

    model = load_trained_whisperformer(args.checkpoint_path, args.num_classes, args.num_decoder_layers, args.num_head_layers, args.device)
    feature_extractor = WhisperFeatureExtractor.from_pretrained("openai/whisper-small", local_files_only=True)

    all_labels = {"onset": [], "offset": [], "cluster": [], "quality": []}
    all_preds_final  = {"onset": [], "offset": [], "cluster": [], "score": []}

    audio_path_list_val, label_path_list_val = get_audio_and_label_paths_from_folders(
        args.audio_folder, args.label_folder)

    audio_list_val, label_list_val = load_data(audio_path_list_val, label_path_list_val, cluster_codebook = cluster_codebook, n_threads = 1 )

    audio_list_val, label_list_val, metadata_list = slice_audios_and_labels( audio_list_val, label_list_val, args.total_spec_columns )
    print(f"Created {len(audio_list_val)} validation samples after slicing")

    # Create validation dataloader
    val_dataset = WhisperFormerDatasetQuality(audio_list_val, label_list_val, args.total_spec_columns, feature_extractor, args.num_classes, args.low_quality_value)
    val_dataloader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False,
                                    num_workers=args.num_workers, collate_fn=collate_fn, drop_last=False)

    preds_by_slice = run_inference_new(
    model=model,
    dataloader=val_dataloader,          # muss mit shuffle=False erstellt sein
    device=args.device,
    threshold=args.threshold,
    iou_threshold=args.iou_threshold,
    metadata_list=metadata_list     # kommt aus slice_audios_and_labels
    )

    final_preds = reconstruct_predictions(
    preds_by_slice=preds_by_slice,
    total_spec_columns=args.total_spec_columns,
    ID_TO_CLUSTER=ID_TO_CLUSTER     # aus datautils importiert
    )
    
    # Predictions pro Datei speichern
    base_name = os.path.splitext(os.path.basename(audio_path))[0]
    json_path = os.path.join(save_dir, f"{base_name}_preds.json")
    with open(json_path, "w") as f:
        json.dump(final_preds, f, indent=2)
    print(f"✅ Predictions saved to {json_path}")
    
        #---- get labels for calculation of F1 val score ----#
    all_labels = {"onset": [], "offset": [], "cluster": [], "quality": []}
    # Labels laden
    for label_path in label_path_list_val:
        with open(label_path, "r") as f:
            labels = json.load(f)
        
        clusters = labels["cluster"]
        labels["cluster"] = [ID_TO_CLUSTER[FIXED_CLUSTER_CODEBOOK[c]] for c in clusters]

        # Quality-Klassen hinzufügen
        if "quality" in labels:
            quality_list = labels["quality"]
        else:
            quality_list = ["unknown"] * len(labels["onset"])


        # --- globale Sammler befüllen ---
        all_labels["onset"].extend(labels["onset"])
        all_labels["offset"].extend(labels["offset"])
        all_labels["cluster"].extend(labels["cluster"])
        all_labels["quality"].extend(quality_list)

    all_preds_final["onset"].extend(final_preds["onset"])
    all_preds_final["offset"].extend(final_preds["offset"])
    all_preds_final["cluster"].extend(final_preds["cluster"])
    all_preds_final["score"].extend(final_preds["score"])

    all_preds, all_labels = group_by_file(all_preds_final, all_labels, metadata_list_val)
    
    tps, fps, fns, fcs, gtps, pps = [],[],[],[],[],[]

    for idx in range(len(all_preds)):

        metrics = evaluate_detection_metrics_with_false_class_qualities(all_labels[idx], all_preds[idx], overlap_tolerance, allowed_qualities = allowed_qualities)
        tps.append(metrics['tp'])
        fps.append(metrics['fp'])
        fns.append(metrics['fn'])
        fcs.append(metrics['fc'])
        gtps.append(metrics['gtp'])
        pps.append(metrics['pp'])
    
    tp_total = sum(tps)
    fp_total = sum(fps)
    fc_total = sum(fcs)
    fn_total = sum(fns)
    gtp_total = sum(gtps)
    pp_total = sum(pps)

    precision = tp_total / (tp_total + fp_total + fc_total) if (tp_total + fp_total + fc_total) > 0 else 0
    recall    = tp_total / (tp_total + fn_total + fc_total) if (tp_total + fn_total + fc_total) > 0 else 0
    f1_all    = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

    metrics_path = os.path.join(save_dir, "metrics_all_qualities.txt")

    with open(metrics_path, "w") as f:
        f.write(f"Globale Metriken für threshold {args.threshold} und iou threshold {args.iou_threshold}: \n")
        f.write(f"TP: {tp_total}\n")
        f.write(f"FP: {fp_total}\n")
        f.write(f"FN: {fn_total}\n")
        f.write(f"FC: {fc_total}\n")
        f.write(f"num gt positives: {gtp_total}\n")
        f.write(f"num predicted positives: {pp_total}\n")
        f.write(f"Precision: {precision:.4f}\n")
        f.write(f"Recall:    {recall:.4f}\n")
        f.write(f"F1-Score:  {f1_all:.4f}\n\n")

    print(f"✅ Globale Metriken gespeichert unter {metrics_path}")

    from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

    # ---- Confusion Matrix für Klassen + None ----
    all_classes = sorted(set(all_labels["cluster"]))   # alle echten Klassen
    class_names = all_classes + ["None"]               # zusätzliche None-Klasse

    y_true = []
    y_pred = []

    matched_labels = set()
    matched_preds = set()

    # Match Events wie bei evaluate_detection_metrics_with_false_class
    for p_idx, (po, pf, pc) in enumerate(zip(all_preds_final['onset'],
                                            all_preds_final['offset'],
                                            all_preds_final['cluster'])):
        for l_idx, (lo, lf, lc, lq) in enumerate(zip(all_labels['onset'],
                                                all_labels['offset'],
                                                all_labels['cluster'],
                                                all_labels['quality'])):
            if l_idx in matched_labels or p_idx in matched_preds or lq == '3':
                continue
            intersection = max(0, min(pf, lf) - max(po, lo))
            union = max(pf, lf) - min(po, lo)
            overlap_ratio = intersection / union if union > 0 else 0
            if overlap_ratio > args.overlap_tolerance:
                matched_labels.add(l_idx)
                matched_preds.add(p_idx)
                # korrekt lokalisiert → check Klassen
                if lc == pc:
                    y_true.append(lc)
                    y_pred.append(pc)
                else:
                    # falsche Klasse
                    y_true.append(lc)
                    y_pred.append(pc)

    # False Negatives (Labels ohne Match)
    for l_idx, (lc, lq) in enumerate(zip(all_labels['cluster'], all_labels['quality'])):
        if l_idx not in matched_labels and lq != '3':
            y_true.append(lc)
            y_pred.append("None")

    # False Positives (Preds ohne Match)
    for p_idx, pc in enumerate(all_preds_final['cluster']):
        if p_idx not in matched_preds:
            y_true.append("None")
            y_pred.append(pc)

    # ---- Confusion Matrix berechnen und plotten ----
    cm = confusion_matrix(y_true, y_pred, labels=class_names)

    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_names)
    fig, ax = plt.subplots(figsize=(8, 6))
    disp.plot(ax=ax, cmap="Blues", xticks_rotation=45)
    plt.title("Confusion-Matrix (Klassen + None)")
    plt.tight_layout()

    cm_path = os.path.join(save_dir, "confusion_matrix_classes_none.png")
    plt.savefig(cm_path, dpi=150)
    plt.close()

    print(f"✅ Confusion-Matrix gespeichert unter {cm_path}")
