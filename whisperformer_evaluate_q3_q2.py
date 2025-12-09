import argparse
import json
import os
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from datetime import datetime
from collections import defaultdict
import numpy as np

from whisperformer_dataset_quality import WhisperFormerDatasetQuality
from whisperformer_model import WhisperFormer
from transformers import WhisperModel, WhisperFeatureExtractor
from datautils import (
    get_audio_and_label_paths_from_folders,
    load_data,
    slice_audios_and_labels,
    FIXED_CLUSTER_CODEBOOK,
    ID_TO_CLUSTER
)
from whisperformer_train import collate_fn, nms_1d_torch, group_by_file
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from sklearn.metrics import precision_score, recall_score, f1_score
import contextlib

cluster_codebook=FIXED_CLUSTER_CODEBOOK
feature_extractor = WhisperFeatureExtractor

def evaluate_detection_metrics_with_false_class_qualities_q3_q2(labels, predictions, overlap_tolerance, allowed_qualities = [1,2,3]):

    label_onsets   = labels['onset']
    label_offsets  = labels['offset']
    label_clusters = labels['cluster']
    label_qualities = labels['quality']

    # only use labels from the allowed quality classes
    if allowed_qualities is not None:
        # try to convert everything to int (if possible)
        try:
            allowed_ints = set(int(q) for q in allowed_qualities)
            label_ints = np.array([int(float(q)) for q in label_qualities])
            mask = np.array([q in allowed_ints for q in label_ints], dtype=bool)
        except ValueError:
            # else use string comparison 
            allowed_str = set(str(q) for q in allowed_qualities)
            qual_str = np.array([str(q) for q in label_qualities])
            mask = np.array([q in allowed_str for q in qual_str], dtype=bool)

        label_onsets   = np.array(label_onsets)[mask]
        label_offsets  = np.array(label_offsets)[mask]
        label_clusters = np.array(label_clusters)[mask]
        label_qualities = np.array(label_qualities)[mask]


    # load predictions
    pred_onsets = np.array(predictions['onset'])
    pred_offsets = np.array(predictions['offset'])
    pred_clusters = np.array(predictions['cluster'])
    pred_scores = np.array(predictions['score'])

    if any(str(x).lower() == "unknown" for x in pred_scores):
        print('Score unknown')
    else:
        # sort predictions by score descending
        order = np.argsort(-pred_scores)
        pred_onsets = pred_onsets[order]
        pred_offsets = pred_offsets[order]
        pred_clusters = pred_clusters[order]
        pred_scores = pred_scores[order]

    matched_labels = set()
    matched_preds = set()
    q2_q3_matched_labels = set()
    q2_q3_matched_preds = set()
    fn_q3_q2 = set()
    false_class = 0

    gtp = len(label_onsets)     
    pp  = len(pred_onsets)

    for p_idx, (po, pf, pc) in enumerate(zip(pred_onsets, pred_offsets, pred_clusters)):
        for l_idx, (lo, lf, lc, lq) in enumerate(zip(label_onsets, label_offsets, label_clusters, label_qualities)):
            if l_idx in matched_labels or p_idx in matched_preds or int(float(lq)) != 1:
                continue
            inter = max(0.0, min(pf, lf) - max(po, lo))
            union = max(pf, lf) - min(po, lo)
            ov = inter / union if union > 0 else 0.0
            if ov > overlap_tolerance and str(pc) == str(lc):
                matched_labels.add(l_idx)
                matched_preds.add(p_idx)
    #dann nach false class
    for p_idx, (po, pf, pc) in enumerate(zip(pred_onsets, pred_offsets, pred_clusters)):
        for l_idx, (lo, lf, lc, lq) in enumerate(zip(label_onsets, label_offsets, label_clusters, label_qualities)):
            if l_idx in matched_labels or p_idx in matched_preds or int(float(lq)) != 1:
                continue
            inter = max(0.0, min(pf, lf) - max(po, lo))
            union = max(pf, lf) - min(po, lo)
            ov = inter / union if union > 0 else 0.0
            if ov > overlap_tolerance and str(pc) != str(lc):
                false_class +=1
                matched_labels.add(l_idx)
                matched_preds.add(p_idx)



    #check for false positives that can be matched with q2 or q3
    for p_idx, (po, pf, pc) in enumerate(zip(pred_onsets, pred_offsets, pred_clusters)):
        for l_idx, (lo, lf, lc, lq) in enumerate(zip(label_onsets, label_offsets, label_clusters, label_qualities)):
            if l_idx in matched_labels  or l_idx in q2_q3_matched_labels or p_idx in matched_preds or int(float(lq)) == 1:
                continue
            inter = max(0.0, min(pf, lf) - max(po, lo))
            union = max(pf, lf) - min(po, lo)
            ov = inter / union if union > 0 else 0.0
            if ov > overlap_tolerance:
                if str(pc) != str(lc):
                    continue
                q2_q3_matched_labels.add(l_idx)
                q2_q3_matched_preds.add(p_idx)

            
    
    #check for false negatives with quality class 3 or 2
    for l_idx, (lo, lf, lc, lq) in enumerate(zip(label_onsets, label_offsets, label_clusters, label_qualities)):
        if l_idx in matched_labels:
            continue
        if lq != '1':
            fn_q3_q2.add(l_idx)
    
    print(len(q2_q3_matched_preds))
    tp = len(matched_labels) - false_class
    fp = len(pred_onsets) - len(matched_preds) -len(q2_q3_matched_preds)
    fn = len(label_onsets) - len(matched_labels) - len(fn_q3_q2) #do not punish false negatives from qc 3 and 2
    fc = false_class

    precision = tp / (tp + fp + fc) if (tp + fp + fc) > 0 else 0.0
    recall    = tp / (tp + fn + fc) if (tp + fn + fc) > 0 else 0.0
    f1        = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

    return {
        'gtp': gtp, 'pp': pp,
        'tp': tp, 'fp': fp, 'fn': fn, 'fc': fc,
        'precision': precision, 'recall': recall, 'f1': f1
    }




def group_by_file(all_preds, all_labels, metadata_list):
    """Gibt dicts zurück: {file_idx: {'onset':[], 'offset':[], 'cluster':[], 'score':[]}}"""
    # Vorhersagen gruppieren
    preds_grouped = defaultdict(lambda: {"onset": [], "offset": [], "cluster": [], "score": []})
    for i, o in enumerate(all_preds["onset"]):
        file_idx = all_preds["orig_idx"][i]  

        preds_grouped[file_idx]["onset"].append(all_preds["onset"][i])
        preds_grouped[file_idx]["offset"].append(all_preds["offset"][i])
        preds_grouped[file_idx]["cluster"].append(all_preds["cluster"][i])
        preds_grouped[file_idx]["score"].append(all_preds["score"][i])

    # Labels gruppieren (falls all_labels noch nicht pro file_idx)
    labels_grouped = defaultdict(lambda: {"onset": [], "offset": [], "cluster": [], "quality": []})
    for i, o in enumerate(all_labels["onset"]):
        file_idx = all_labels["orig_idx"][i]
        labels_grouped[file_idx]["onset"].append(all_labels["onset"][i])
        labels_grouped[file_idx]["offset"].append(all_labels["offset"][i])
        labels_grouped[file_idx]["cluster"].append(all_labels["cluster"][i])
        labels_grouped[file_idx]["quality"].append(all_labels["quality"][i])
    
    return preds_grouped, labels_grouped



# ==================== MAIN ====================

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    #parser.add_argument("--checkpoint_path", required=True)
    parser.add_argument("--audio_folder", required=True)
    parser.add_argument("--label_folder", required=True)
    parser.add_argument("--pred_folder", required=True)
    parser.add_argument("--output_dir", default="inference_outputs")
    parser.add_argument("--total_spec_columns", type=int, default=3000)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--num_classes", type=int, default=3)
    parser.add_argument("--threshold", type=float, default=0)
    parser.add_argument("--iou_threshold", type=float, default=0.4)
    parser.add_argument("--overlap_tolerance", type=float, default=0.3)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--num_decoder_layers", type = int, default = 3)
    parser.add_argument("--num_head_layers", type = int, default = 2)
    parser.add_argument("--low_quality_value", type = float, default = 0.5)
    parser.add_argument("--value_q2", type = float, default = 1)
    parser.add_argument("--centerframe_size", type = float, default = 0.6)
    parser.add_argument("--allowed_qualities", nargs='+', type=int, default=[1,2,3])    
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


    all_labels = {"onset": [], "offset": [], "cluster": [], "quality": []}
    all_preds  = {"onset": [], "offset": [], "cluster": [], "score": [], 'orig_idx': []}

    #for labels
    audio_path_list_val, label_path_list_val = get_audio_and_label_paths_from_folders(
        args.audio_folder, args.label_folder)
    audio_list_val, label_list_val = load_data(audio_path_list_val, label_path_list_val, cluster_codebook = cluster_codebook, n_threads = 1 )

    audio_list_val, label_list_val, metadata_list_val = slice_audios_and_labels( audio_list_val, label_list_val, args.total_spec_columns )

    # Create validation dataloader
    val_dataset = WhisperFormerDatasetQuality(audio_list_val, label_list_val, args.total_spec_columns, feature_extractor, args.num_classes, args.low_quality_value, args.centerframe_size, args.value_q2)
    val_dataloader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False,
                                    num_workers=args.num_workers, collate_fn=collate_fn, drop_last=False)

    
    all_labels = {"onset": [], "offset": [], "cluster": [], "quality": [], "orig_idx": []}
    # Labels laden
    for i, label_path in enumerate(label_path_list_val):
        stem = os.path.basename(label_path).split('.')[0]
        print(f'label stem: {stem}')
        with open(label_path, "r") as f:
            labels = json.load(f)
        
        clusters = labels["cluster"]
        labels["cluster"] = [ID_TO_CLUSTER[FIXED_CLUSTER_CODEBOOK[c]] for c in clusters]
        labels['orig_idx'] = [stem]*len(labels["cluster"])
        

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
        all_labels["orig_idx"].extend(labels['orig_idx'])



 #---- get predictions for calculation of F1 val score ----#
    # Labels laden
    folder_path = args.pred_folder

    # Alle JSON-Dateien auflisten und sortieren
    json_files = sorted([f for f in os.listdir(folder_path) if f.endswith((".json", ".jsonr")) and f != "run_arguments.json"])

    # Dateien nacheinander laden
    all_data = []

    for i, file_name in enumerate(json_files):
        stem = os.path.basename(file_name).split('.')[0]
        print(f'pred stem: {stem}')
        file_path = os.path.join(folder_path, file_name)
        with open(file_path, "r") as f:
            try:
                preds = json.load(f)
            except json.JSONDecodeError:
                print(f"[WARN] Datei {file_name} ist keine gültige JSON-Datei — wird übersprungen.")
                continue

        # Prüfen, ob Datei die erwarteten Keys enthält
        required_keys = ("onset", "offset", "cluster")
        if not all(k in preds for k in required_keys):
            print(f"[WARN] Datei {file_name} hat nicht das erwartete Format — wird übersprungen.")
            continue

        # Cluster umwandeln
        clusters = preds["cluster"]
        preds["cluster"] = [ID_TO_CLUSTER[FIXED_CLUSTER_CODEBOOK[c]] for c in clusters]

        # Alle Segmente dieser Datei bekommen denselben orig_idx
        preds["orig_idx"] = [stem] * len(preds["cluster"])

        # Quality-Klassen hinzufügen
        if "score" in preds:
            score_list = preds["score"]
        else:
            score_list = ["unknown"] * len(preds["onset"])
        
        #only preds above threshold
        if "score" in preds:
            filtered = [
                (on, off, cl, sc, oi)
                for on, off, cl, sc, oi in zip(
                    preds["onset"],
                    preds["offset"],
                    preds["cluster"],
                    preds["score"],
                    preds["orig_idx"]
                )
                if float(sc) > args.threshold
            ]
            if not filtered:
                continue
            preds["onset"], preds["offset"], preds["cluster"], preds["score"], preds["orig_idx"] = zip(*filtered)
        
        # --- globale Sammler befüllen ---
        all_preds["onset"].extend(preds["onset"])
        all_preds["offset"].extend(preds["offset"])
        all_preds["cluster"].extend(preds["cluster"])
        all_preds["score"].extend(score_list)
        all_preds["orig_idx"].extend(preds["orig_idx"])

    all_preds_grouped, all_labels_grouped = group_by_file(all_preds, all_labels, metadata_list_val)
    
    tps, fps, fns, fcs, gtps, pps = [],[],[],[],[],[]
    
    for file_idx in all_preds_grouped:

        metrics = evaluate_detection_metrics_with_false_class_qualities_q3_q2(all_labels_grouped[file_idx], all_preds_grouped[file_idx], overlap_tolerance = args.overlap_tolerance, allowed_qualities = args.allowed_qualities)
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

 