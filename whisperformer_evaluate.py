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
from whisperformer_train import collate_fn, nms_1d_torch, group_by_file, evaluate_detection_metrics_with_false_class_qualities
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from sklearn.metrics import precision_score, recall_score, f1_score
import contextlib

feature_extractor = WhisperFeatureExtractor
cluster_codebook=FIXED_CLUSTER_CODEBOOK



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
    parser.add_argument("--threshold", type=float, default=0.4)
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
        all_preds["score"].extend(preds["score"])
        all_preds["orig_idx"].extend(preds["orig_idx"])

    all_preds_grouped, all_labels_grouped = group_by_file(all_preds, all_labels, metadata_list_val)
    
    tps, fps, fns, fcs, gtps, pps = [],[],[],[],[],[]
    all_file_ids = set(all_labels_grouped.keys())

    for file_idx in all_file_ids:
        preds = all_preds_grouped.get(file_idx, {"onset": [], "offset": [], "cluster": [], "score": []})
        labels = all_labels_grouped[file_idx]
    #for file_idx in all_preds_grouped:

        metrics = evaluate_detection_metrics_with_false_class_qualities(all_labels_grouped[file_idx], all_preds_grouped[file_idx], overlap_tolerance = args.overlap_tolerance, allowed_qualities = args.allowed_qualities)
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

    # ---- Confusion Matrix für Klassen + None ----
    allowed_set = set(args.allowed_qualities)

    filtered_labels = {
        "onset": [],
        "offset": [],
        "cluster": [],
        "quality": [],
    }

    for on, off, cl, q in zip(all_labels["onset"], all_labels["offset"], all_labels["cluster"], all_labels["quality"]):
        # Nur gewünschte Qualitäten behalten
        if int(float(q)) in allowed_set:
            filtered_labels["onset"].append(on)
            filtered_labels["offset"].append(off)
            filtered_labels["cluster"].append(cl)
            filtered_labels["quality"].append(q)


    all_classes = sorted(set(filtered_labels["cluster"]))   # alle echten Klassen
    class_names = all_classes + ["None"]               # zusätzliche None-Klasse

    y_true = []
    y_pred = []

    matched_labels = set()
    matched_preds = set()

    all_fn_qualities = []
    
    # Match Events wie bei evaluate_detection_metrics_with_false_class
    for p_idx, (po, pf, pc) in enumerate(zip(all_preds['onset'],
                                            all_preds['offset'],
                                            all_preds['cluster'])):
        for l_idx, (lo, lf, lc, lq) in enumerate(zip(filtered_labels['onset'],
                                                filtered_labels['offset'],
                                                filtered_labels['cluster'],
                                                filtered_labels['quality'])):
            if l_idx in matched_labels or p_idx in matched_preds:
                continue
            intersection = max(0, min(pf, lf) - max(po, lo))
            union = max(pf, lf) - min(po, lo)
            overlap_ratio = intersection / union if union > 0 else 0
            if overlap_ratio > args.overlap_tolerance and str(pc) == str(lc):
                matched_labels.add(l_idx)
                matched_preds.add(p_idx)
                y_true.append(lc)
                y_pred.append(pc)

    for p_idx, (po, pf, pc) in enumerate(zip(all_preds['onset'],
                                            all_preds['offset'],
                                            all_preds['cluster'])):
        for l_idx, (lo, lf, lc, lq) in enumerate(zip(filtered_labels['onset'],
                                                filtered_labels['offset'],
                                                filtered_labels['cluster'],
                                                filtered_labels['quality'])):
            if l_idx in matched_labels or p_idx in matched_preds:
                continue
            intersection = max(0, min(pf, lf) - max(po, lo))
            union = max(pf, lf) - min(po, lo)
            overlap_ratio = intersection / union if union > 0 else 0
            if overlap_ratio > args.overlap_tolerance and str(pc) != str(lc):
                matched_labels.add(l_idx)
                matched_preds.add(p_idx)
                y_true.append(lc)
                y_pred.append(pc)


    # False Negatives
    for l_idx, (lc, lq) in enumerate(zip(filtered_labels['cluster'], filtered_labels['quality'])):
        if l_idx not in matched_labels:
            all_fn_qualities.append(lq)
            y_true.append(lc)
            y_pred.append("None")
        

    # False Positives 
    for p_idx, pc in enumerate(all_preds['cluster']):
        if p_idx not in matched_preds:
            y_true.append("None")
            y_pred.append(pc)

    # ---- Confusion Matrix berechnen und plotten ----
    new_labels = ["hmm", "moan", "wail", "None"]

    cm = confusion_matrix(y_true, y_pred, labels=["h", "m", "w", "None"])
    """
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=new_labels)
    fig, ax = plt.subplots(figsize=(8, 6))

    im = disp.plot(ax=ax, cmap="Blues", xticks_rotation=45).im_
    cbar = im.colorbar
    cbar.ax.tick_params(labelsize=16)

    disp.plot(ax=ax, cmap="Blues", xticks_rotation=45)

    # Schriftgrößen anpassen
    ax.tick_params(axis='both', labelsize=16)     # Achsenbeschriftung
    ax.set_xlabel(ax.get_xlabel(), fontsize=18)   # x-Label
    ax.set_ylabel(ax.get_ylabel(), fontsize=18)   # y-Label
    #ax.set_title("Confusion Matrix", fontsize=14) # optional

    # Zahlen in der Matrix anpassen:
    for text in disp.text_.ravel():               # disp.text_ enthält alle Zellen
        text.set_fontsize(16)

    plt.tight_layout()

    cm_path = os.path.join(save_dir, "confusion_matrix_classes_none.png")
    plt.savefig(cm_path, dpi=300)
    plt.close()
    """
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=new_labels)
    fig, ax = plt.subplots(figsize=(8, 6))

    # Plotten + Image-Objekt abgreifen
    plot_obj = disp.plot(ax=ax, cmap="Blues", xticks_rotation=45)
    im = plot_obj.im_                # Das Image der Confusion Matrix
    cbar = im.colorbar               # Die Colorbar bekommen

    # Schriftgröße der Colorbar
    cbar.ax.tick_params(labelsize=16)

    # Schriftgrößen anpassen
    ax.tick_params(axis='both', labelsize=16)
    ax.set_xlabel(ax.get_xlabel(), fontsize=18)
    ax.set_ylabel(ax.get_ylabel(), fontsize=18)

    # Schriftgröße der Zellen
    for text in disp.text_.ravel():
        text.set_fontsize(16)

    plt.tight_layout()

    cm_path = os.path.join(save_dir, "confusion_matrix_classes_none.png")
    plt.savefig(cm_path, dpi=300)
    plt.close()

