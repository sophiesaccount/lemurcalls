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
from wf_soft_large import collate_fn, nms_1d_torch, group_by_file, evaluate_detection_metrics_with_false_class_qualities
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from sklearn.metrics import precision_score, recall_score, f1_score
import contextlib
from whisperformer_evaluate_q3_q2 import evaluate_detection_metrics_with_false_class_qualities_q3_q2

cluster_codebook=FIXED_CLUSTER_CODEBOOK
feature_extractor = WhisperFeatureExtractor


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

    label_stems = [os.path.basename(label_path).split('.')[0] for label_path in label_path_list_val]
    stem_to_idx = {stem: i for i, stem in enumerate(label_stems)}
    print(f'label_stems: {label_stems}')


    for i, label_path in enumerate(label_path_list_val):
        #stem = os.path.basename(label_path).split('.')[0]
        #print(f'label stem: {stem}')
        with open(label_path, "r") as f:
            labels = json.load(f)

        label_stem = os.path.basename(label_path).split('.')[0]
        orig_idx = stem_to_idx[label_stem]

        clusters = labels["cluster"]
        labels["cluster"] = [ID_TO_CLUSTER[FIXED_CLUSTER_CODEBOOK[c]] for c in clusters]
        #labels['orig_idx'] = [stem]*len(labels["cluster"])
        

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
        #all_labels["orig_idx"].extend(labels['orig_idx'])
        #all_labels["orig_idx"].extend([i for _ in range(len(labels["onset"]))])
        #all_labels["orig_idx"].extend([i for _ in range(len(labels["onset"]))])
        all_labels["orig_idx"].extend([orig_idx] * len(labels["onset"]))


    precisions, recalls, f1s, f2s = [], [], [], []
    #thresholds = np.arange(0, 1.02, 0.02)
    thresholds = [0.36]
    for threshold in thresholds:
            all_preds  = {"onset": [], "offset": [], "cluster": [], "score": [], 'orig_idx': []}

        #---- get predictions for calculation of F1 val score ----#
            # Labels laden
            folder_path = args.pred_folder

            # Alle JSON-Dateien auflisten und sortieren
            json_files = sorted([f for f in os.listdir(folder_path) if f.endswith((".json", ".jsonr")) and f != "run_arguments.json"])

            # Dateien nacheinander laden
            all_data = []

            for i, file_name in enumerate(json_files):
                #stem = os.path.basename(file_name).split('.')[0]
                #print(f'pred stem: {stem}')
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

                pred_stem = os.path.basename(file_name).split('.')[0]
                print(f'pred_stem: {pred_stem}')

                if pred_stem not in stem_to_idx:
                    print(f"[WARN] Kein Matching für Prediction {file_name} – wird übersprungen.")
                    continue
                orig_idx = stem_to_idx[pred_stem]

                # Quality-Klassen hinzufügen
                if "score" in preds:
                    score_list = preds["score"]
                else:
                    score_list = ["unknown"] * len(preds["onset"])
                
                #only preds above threshold
                if "score" in preds:
                    filtered = [
                        (on, off, cl, sc)
                        for on, off, cl, sc in zip(
                            preds["onset"],
                            preds["offset"],
                            preds["cluster"],
                            preds["score"],
                        )
                        if float(sc) > threshold
                    ]
                    if not filtered:
                        continue
                    preds["onset"], preds["offset"], preds["cluster"], preds["score"] = zip(*filtered)
                
                # --- globale Sammler befüllen ---
                all_preds["onset"].extend(preds["onset"])
                all_preds["offset"].extend(preds["offset"])
                all_preds["cluster"].extend(preds["cluster"])
                all_preds["score"].extend(preds["score"])
                #all_preds["orig_idx"].extend(preds["orig_idx"])
                all_preds["orig_idx"].extend([orig_idx] * len(preds["onset"]))


            all_preds_grouped, all_labels_grouped = group_by_file(all_preds, all_labels, metadata_list_val)
            
            tps, fps, fns, fcs, gtps, pps = [],[],[],[],[],[]
            
            #all_file_ids = set(all_labels_grouped.keys())
            #all_file_ids = set(all_labels_grouped.keys()) | set(all_preds_grouped.keys())

            #for file_idx in all_file_ids:
            for file_idx in range(38):
                print(file_idx)

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

            f2 = (5 * precision * recall) / (4 * precision + recall) if (4 * precision + recall) > 0 else 0


        
            precisions.append(precision)
            recalls.append(recall)
            f1s.append(f1_all)
            f2s.append(f2)
    
    best_idx = np.argmax(f1s)

    best_precision = precisions[best_idx]
    best_recall = recalls[best_idx]
    best_f1 = f1s[best_idx]
    best_th = thresholds[best_idx]

    metrics_path = os.path.join(save_dir, "metrics_all_qualities.txt")

    with open(metrics_path, "w") as f:
        f.write(f"Globale Metriken für threshold {args.threshold} und iou threshold {args.iou_threshold}: \n")
        f.write(f"TP: {tp_total}\n")
        f.write(f"FP: {fp_total}\n")
        f.write(f"FN: {fn_total}\n")
        f.write(f"FC: {fc_total}\n")
        f.write(f"num gt positives: {gtp_total}\n")
        f.write(f"num predicted positives: {pp_total}\n")
        f.write(f"Thresholds: {thresholds}\n")
        precisions_rounded = [f"{p:.4f}" for p in precisions]
        recalls_rounded    = [f"{r:.4f}" for r in recalls]
        f1s_rounded        = [f"{f1:.4f}" for f1 in f1s]

        f.write(f"Precision: {precisions_rounded}\n")
        f.write(f"Recall:    {recalls_rounded}\n")
        f.write(f"F1-Score:  {f1s_rounded}\n\n")
    print(f"✅ Globale Metriken gespeichert unter {metrics_path}")

    import matplotlib.pyplot as plt

    plt.figure(figsize=(6, 5))
    plt.plot(recalls, precisions, marker="o")
    # 🔴 Besten F1 Punkt markieren
    plt.scatter(best_recall, best_precision, color="red", s=80, zorder=5, label="Best F1")

    # 🔴 Text leicht oberhalb platzieren
    plt.text(
        best_recall,
        best_precision + 0.02,
        f"Best F1 = {best_f1:.3f} for threshold {best_th}",
        color="red",
        fontsize=10,
        ha="center"
    )
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title("Precision–Recall Curve")
    plt.grid(True)

    file_path = os.path.join(save_dir, "precision_recall_curve.png")

    plt.savefig(file_path, dpi=300, bbox_inches="tight")


    # === Plot F1 vs Threshold ===
    plt.figure(figsize=(7, 5))

    # F1 plotten
    plt.plot(thresholds, f1s, marker="o", label="F1 Score")
    # F2 plotten
    plt.plot(thresholds, f2s, marker="s", label="F2 Score")

    # Jeden 5. Punkt beschriften
    for i in range(0, len(thresholds), 5):
        th = thresholds[i]

        # F1 beschriften
        plt.text(
            th, f1s[i] + 0.01,
            f"{th:.2f}",
            fontsize=7,
            ha="center",
            color="black"
        )

        # F2 beschriften, etwas tiefer setzen
        plt.text(
            th, f2s[i] - 0.02,
            f"{th:.2f}",
            fontsize=7,
            ha="center",
            color="gray"
        )

    # Beste F1 markieren
    best_idx_f1 = np.argmax(f1s)
    plt.scatter(thresholds[best_idx_f1], f1s[best_idx_f1], color="red", s=80, zorder=5)
    plt.text(
        thresholds[best_idx_f1], f1s[best_idx_f1] + 0.03,
        f"Best F1 {f1s[best_idx_f1]:.3f}",
        color="red", fontsize=9, ha="center"
    )

    # Beste F2 markieren
    best_idx_f2 = np.argmax(f2s)
    plt.scatter(thresholds[best_idx_f2], f2s[best_idx_f2], color="blue", s=80, zorder=5)
    plt.text(
        thresholds[best_idx_f2], f2s[best_idx_f2] + 0.03,
        f"Best F2 {f2s[best_idx_f2]:.3f}",
        color="blue", fontsize=9, ha="center"
    )

    plt.xlabel("Threshold")
    plt.ylabel("Score")
    plt.title("F1 & F2 Scores vs Threshold")
    plt.grid(True)
    plt.legend()

    outfile = os.path.join(save_dir, "f1_f2_vs_threshold.png")
    plt.savefig(outfile, dpi=300, bbox_inches="tight")
    plt.close()
