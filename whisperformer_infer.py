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
from whisperformer_train import collate_fn, nms_1d_torch, evaluate_detection_metrics_with_false_class_qualities, group_by_file
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from sklearn.metrics import precision_score, recall_score, f1_score
import contextlib



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

def run_inference_new(model, dataloader, device, threshold, iou_threshold, metadata_list):
    preds_by_slice = []
    slice_idx = 0
    model.eval()

    with torch.no_grad():
        for batch in dataloader:
            for k, v in batch.items():
                if isinstance(v, torch.Tensor):
                    batch[k] = v.to(device, non_blocking=True)

            use_autocast = (
                isinstance(device, str) and device.startswith("cuda")
            ) or (
                hasattr(device, "type") and device.type == "cuda"
            )
            autocast_ctx = torch.amp.autocast(
                device_type="cuda", dtype=torch.float16
            ) if use_autocast else contextlib.nullcontext()

            with autocast_ctx:
                class_preds, regr_preds = model(batch["input_features"])
                class_probs = torch.sigmoid(class_preds)

            B, T, C = class_preds.shape

            for b in range(B):
                meta = metadata_list[slice_idx]
                slice_idx += 1

                ############################################################
                # 1) ALLE Intervalle aller Klassen sammeln (NEU)
                ############################################################
                all_intervals = []     ### ADDED
                for c in range(C):
                    for t in range(T):
                        score = class_probs[b, t, c]
                        if float(score) > threshold:
                            start = t - regr_preds[b, t, 0]
                            end   = t + regr_preds[b, t, 1]

                            all_intervals.append(
                                torch.tensor([start, end, score, c], device=score.device)
                            )  ### ADDED

                # Wenn keine Intervalle -> leer zurückgeben
                if len(all_intervals) == 0:
                    preds_per_class = [{"class": c, "intervals": []} for c in range(C)]
                    preds_by_slice.append({
                        "original_idx": meta["original_idx"],
                        "segment_idx": meta["segment_idx"],
                        "preds": preds_per_class
                    })
                    continue

                all_intervals = torch.stack(all_intervals)  ### ADDED

                ############################################################
                # 2) Klassenübergreifendes NMS (NEU)
                ############################################################
                kept = soft_nms_1d_torch(all_intervals, iou_threshold)   ### ADDED

                ############################################################
                # 3) Wieder pro Klasse einsortieren (NEU)
                ############################################################
                preds_per_class = []  ### ADDED
                for c in range(C):    ### ADDED
                    mask = kept[:, 3] == c
                    cls_int = kept[mask][:, :3].cpu().tolist()   # class-id weg
                    preds_per_class.append({"class": c, "intervals": cls_int})

                ############################################################
                # Rückgabe wie vorher
                ############################################################
                preds_by_slice.append({
                    "original_idx": meta["original_idx"],
                    "segment_idx": meta["segment_idx"],
                    "preds": preds_per_class
                })

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
    parser.add_argument("--value_q2", type = float, default = 1)
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

    all_preds_final  = {"onset": [], "offset": [], "cluster": [], "score": [], "orig_idx": []}
    for audio_path, label_path in zip(audio_paths, label_paths):
        print(f"\n===== Processing {os.path.basename(audio_path)} =====")
        audio_list, label_list = load_data([audio_path], [label_path], cluster_codebook=cluster_codebook, n_threads=1)
        audio_list, label_list, metadata_list = slice_audios_and_labels(audio_list, label_list, args.total_spec_columns)

        dataset = WhisperFormerDatasetQuality(audio_list, label_list, args.total_spec_columns, feature_extractor, args.num_classes, args.low_quality_value, args.value_q2,args.centerframe_size)
        dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False,
                                collate_fn=collate_fn, drop_last=False)

        preds_by_slice = run_inference_new(
        model=model,
        dataloader=dataloader,          # muss mit shuffle=False erstellt sein
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


    