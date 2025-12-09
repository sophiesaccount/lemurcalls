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
from whisperformer_train import collate_fn, nms_1d_torch, evaluate_detection_metrics_with_false_class_qualities_q3, group_by_file
from whisperformer_evaluate_q3_q2 import evaluate_detection_metrics_with_false_class_qualities_q3_q2
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from sklearn.metrics import precision_score, recall_score, f1_score
import contextlib

cluster_codebook=FIXED_CLUSTER_CODEBOOK
feature_extractor = WhisperFeatureExtractor

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
    parser.add_argument("--iou_threshold", type=float, default=1)
    parser.add_argument("--overlap_tolerance", type=float, default=0.01)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--num_decoder_layers", type = int, default = 3)
    parser.add_argument("--num_head_layers", type = int, default = 2)
    parser.add_argument("--low_quality_value", type = float, default = 0.5)
    parser.add_argument("--value_q2", type = float, default = 1)
    parser.add_argument("--centerframe_size", type = float, default = 0.6)
    parser.add_argument("--allowed_qualities", default=[1,2,3], nargs='+', type=int)
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
                if sc > args.threshold
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

        # ==== False Positives mit Quality = 3 analysieren ====

    # Wir wollen prüfen, ob die vorhergesagten Events (FPs)
    # eventuell mit GT-Events überlappen, die Quality 3 haben.

    fp_from_quality3 = 0
    total_fp_checked = 0

    for file_idx in all_preds_grouped:
        preds = all_preds_grouped[file_idx]
        labels = all_labels_grouped[file_idx]

        for po, pf, pc in zip(preds["onset"], preds["offset"], preds["cluster"]):
            # Zuerst prüfen, ob diese Vorhersage ein FP war
            # (d.h. sie wurde NICHT als TP oder FC gezählt)
            is_fp = True
            for lo, lf, lc, lq in zip(labels["onset"], labels["offset"], labels["cluster"], labels["quality"]):
                # Prüfe Überschneidung
                intersection = max(0, min(pf, lf) - max(po, lo))
                union = max(pf, lf) - min(po, lo)
                overlap_ratio = intersection / union if union > 0 else 0

                if overlap_ratio > args.overlap_tolerance:
                    # Es gibt eine Überlappung — wenn lq != 3, dann ist das eigentlich ein Match → kein FP
                    if int(lq) in args.allowed_qualities:
                        is_fp = False
                        continue

            if is_fp:
                total_fp_checked += 1

                # Jetzt prüfen, ob dieser FP evtl. mit einem Quality-3 Label überlappt
                for lo, lf, lc, lq in zip(labels["onset"], labels["offset"], labels["cluster"], labels["quality"]):
                    intersection = max(0, min(pf, lf) - max(po, lo))
                    union = max(pf, lf) - min(po, lo)
                    overlap_ratio = intersection / union if union > 0 else 0
                    if overlap_ratio > args.overlap_tolerance and int(lq) == 3:
                        fp_from_quality3 += 1
                        break

    print(f"🔍 False Positives insgesamt: {total_fp_checked}")
    if total_fp_checked > 0:
        print(f"🟠 Davon stammen {fp_from_quality3} aus Quality=3 Labels "
            f"({fp_from_quality3 / total_fp_checked * 100:.1f} %)")




    from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

    # ---- Confusion Matrix für Klassen + None ----
    all_classes = sorted(set(all_labels["cluster"]))   # alle echten Klassen
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
        for l_idx, (lo, lf, lc, lq) in enumerate(zip(all_labels['onset'],
                                                all_labels['offset'],
                                                all_labels['cluster'],
                                                all_labels['quality'])):
            if l_idx in matched_labels or p_idx in matched_preds:
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
        if l_idx not in matched_labels:
            all_fn_qualities.append(lq)
            y_true.append(lc)
            y_pred.append("None")
        

    # False Positives (Preds ohne Match)
    for p_idx, pc in enumerate(all_preds['cluster']):
        if p_idx not in matched_preds:
            y_true.append("None")
            y_pred.append(pc)



    # === Precision-Recall-Kurven für verschiedene IoU-Thresholds ===
    overlap_tols = [0.1, 0.2, 0.3, 0.4, 0.5]
    thresholds = np.arange(0.1, 1.0, 0.05)
    #thresholds = [0.4, 0.475, 0.5]

    plt.figure(figsize=(8, 6))

    for overlap_tol in overlap_tols:
        precision_list = []
        recall_list = []
        f1_list = []

        for thr in thresholds:
            # Filtere Predictions nach Score-Threshold
            filtered_preds = {
                "onset": [],
                "offset": [],
                "cluster": [],
                "score": [],
                "orig_idx": []
            }

            for po, pf, pc, ps, oi in zip(
                all_preds["onset"],
                all_preds["offset"],
                all_preds["cluster"],
                all_preds["score"],
                all_preds["orig_idx"]
            ):
                if ps >= thr:
                    filtered_preds["onset"].append(po)
                    filtered_preds["offset"].append(pf)
                    filtered_preds["cluster"].append(pc)
                    filtered_preds["score"].append(ps)
                    filtered_preds["orig_idx"].append(oi)

            filtered_grouped, _ = group_by_file(filtered_preds, all_labels, metadata_list_val)

            tps, fps, fns, fcs = [], [], [], []

            for file_idx in filtered_grouped:
                metrics = evaluate_detection_metrics_with_false_class_qualities_q3(
                    all_labels_grouped[file_idx],
                    filtered_grouped[file_idx],
                    overlap_tolerance=overlap_tol,  # <--- hier iou_thr verwenden!
                    allowed_qualities=args.allowed_qualities
                )
                tps.append(metrics["tp"])
                fps.append(metrics["fp"])
                fns.append(metrics["fn"])
                fcs.append(metrics["fc"])

            tp_total = sum(tps)
            fp_total = sum(fps)
            fn_total = sum(fns)
            fc_total = sum(fcs)

            precision = tp_total / (tp_total + fp_total + fc_total) if (tp_total + fp_total + fc_total) > 0 else 0
            recall = tp_total / (tp_total + fn_total + fc_total) if (tp_total + fn_total + fc_total) > 0 else 0
            #precision = (tp_total + fc_total)/ (tp_total + fc_total + fp_total) if (tp_total + fp_total) > 0 else 0
            #recall = (tp_total + fc_total) / (tp_total + fc_total + fn_total) if (tp_total + fn_total) > 0 else 0
            f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

            precision_list.append(precision)
            recall_list.append(recall)
            f1_list.append(f1)

        #calculate best f1
        best_f1 = np.max(f1_list)
        best_threshold = thresholds[np.argmax(f1_list)]
        print(f'for overlap_tol = {overlap_tol}: best f1 is {best_f1} for threshold {best_threshold}')
        # === Zeichne Kurve für diesen IOU-Threshold ===
        plt.plot(
            recall_list,
            precision_list,
            marker='o',
            label=f'Overlap Toelrance={overlap_tol:.2f}'
        )

        # === Beschrifte jeden Punkt mit Threshold-Wert ===
        for r, p, t in zip(recall_list, precision_list, thresholds):
            plt.text(r, p, f"{t:.2f}", fontsize=7, ha='right', va='bottom')

    # === Plot konfigurieren ===
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title("Precision-Recall-Kurven für verschiedene IoU Thresholds")
    plt.grid(True)
    plt.legend(title="Overlap Tolerance (IoU)")
    plt.tight_layout()

    # === Speichern ===
    pr_curve_path = os.path.join(save_dir, "precision_recall_curve_multi_iou.png")
    plt.savefig(pr_curve_path, dpi=150)
    plt.close()

    print(f"✅ Precision-Recall-Kurven gespeichert unter {pr_curve_path}")
