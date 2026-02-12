import argparse
import json
import os
import torch
from torch.utils.data import DataLoader
from datetime import datetime
from collections import defaultdict
import numpy as np
from copy import deepcopy

from .dataset import WhisperFormerDatasetQuality
from .model import WhisperFormer
from transformers import WhisperModel, WhisperFeatureExtractor, WhisperConfig
from ..datautils import (
    get_audio_and_label_paths_from_folders,
    load_data,
    slice_audios_and_labels,
    FIXED_CLUSTER_CODEBOOK,
    ID_TO_CLUSTER
)
from .train import collate_fn, run_inference_new


def slice_audios_and_labels(audio_list, label_list, total_spec_columns, pad_columns=2000, offset=0):

    """
    Slice audios into overlapping segments with different offsets (0, 1/3, 2/3).
    Returns expanded audio_list, label_list, metadata_list with offset_frac.
    """
    padded_audio_list = [
        np.concatenate([
            np.zeros((pad_columns, 1)),           # Padding vorne
            np.expand_dims(a, axis=1),            # a von (T,) → (T,1)
            np.zeros((pad_columns, 1))            # Padding hinten
        ], axis=0)
        for a in audio_list
    ]
    new_audios, new_labels, new_metadata = [], [], []
    sec_per_col = 0.01
    spec_time_step = 0.01
    total_spec_columns=3000

    for orig_idx, (audio, label) in enumerate(zip(audio_list, label_list)):
        sr = 16000
        clip_duration = total_spec_columns * spec_time_step
        num_samples_in_clip = int(round(clip_duration * sr))

        start = offset
        seg_idx = 0

        while start < len(audio):
            end = start + num_samples_in_clip
            audio_clip = audio[start:end]

            #if len(audio_clip) < sr * 0.1:  # skip super short
            #    break


            # Labels anpassen: nur Events im Zeitfenster behalten
            start_time = start / sr
            end_time = end / sr
            intersected_indices = np.logical_and(
                label["onset"] < end_time, label["offset"] > start_time
            )

            label_clip = deepcopy(label)
            label_clip.update({
                "onset": np.maximum(label["onset"][intersected_indices], start_time) - start_time,
                "offset": np.minimum(label["offset"][intersected_indices], end_time) - start_time,
                "cluster_id": label["cluster_id"][intersected_indices],
                "cluster": [label["cluster"][idx] for idx in np.argwhere(intersected_indices)[:, 0]],
                "quality": [label["quality"][idx] for idx in np.argwhere(intersected_indices)[:, 0]]
            })

            # speichern
            new_audios.append(audio_clip)
            new_labels.append(label_clip)
            new_metadata.append({
                "original_idx": orig_idx,
                "segment_idx": seg_idx,
                "offset_frac": offset,   # tells us in which trial we are, by giving us the offset
                "trial_id": offset
            })

            start += num_samples_in_clip
            seg_idx += 1


    return new_audios, new_labels, new_metadata

# ==================== MODEL LOADING ====================

def detect_whisper_size_from_state_dict(state_dict):
    """
    Whisper-Größe anhand der Gewicht-Dimensionen im State Dict erkennen.
    Whisper Base: d_model=512, Whisper Large: d_model=1280
    """
    for key in state_dict.keys():
        if "conv1.weight" in key:
            d_model = state_dict[key].shape[0]
            if d_model == 1280:
                return "large"
            elif d_model == 512:
                return "base"
    return None


def load_trained_whisperformer(checkpoint_path, num_classes, num_decoder_layers, num_head_layers, device, whisper_size=None):
    """
    Lade trainiertes WhisperFormer Modell.
    
    Der Checkpoint enthält ALLE Gewichte (inkl. frozen Encoder).
    Es wird nur die WhisperConfig (kleines JSON) geladen, NICHT das volle
    Pretrained-Modell -- die Encoder-Gewichte kommen direkt aus dem Checkpoint.
    
    Whisper-Größe wird automatisch aus dem Checkpoint erkannt
    (d_model=512 -> base, d_model=1280 -> large).
    """
    # 1) State Dict einmal laden
    state_dict = torch.load(checkpoint_path, map_location=device)
    
    # 2) Whisper-Größe bestimmen
    if whisper_size and whisper_size.lower() in ["base", "large"]:
        detected_size = whisper_size.lower()
    else:
        detected_size = detect_whisper_size_from_state_dict(state_dict)
        if detected_size is None:
            # Fallback: aus Checkpoint-Pfad erkennen
            path_lower = checkpoint_path.lower()
            if "large" in path_lower:
                detected_size = "large"
            else:
                detected_size = "base"
    
    print(f"Detected Whisper size: {detected_size}")
    
    # 3) Nur die Config laden (kleines JSON) -- kein from_pretrained noetig,
    #    da alle Gewichte (inkl. Encoder) aus dem Checkpoint kommen
    _project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
    _whisper_models_dir = os.path.join(_project_root, "whisper_models")
    config_path = os.path.join(_whisper_models_dir, f"whisper_{detected_size}")
    
    config = WhisperConfig.from_pretrained(config_path)
    whisper_model = WhisperModel(config)  # leere Gewichte, kein Download
    encoder = whisper_model.encoder
    
    model = WhisperFormer(encoder, num_classes=num_classes, num_decoder_layers=num_decoder_layers, num_head_layers=num_head_layers)
    
    # 4) Praefix-Korrektur: Checkpoint hat "encoder.X", Modell erwartet "encoder.encoder.X"
    #    wegen WhisperEncoder-Wrapper (self.encoder = encoder in whisperformer_model_base.py)
    needs_remap = any(
        k.startswith("encoder.") and not k.startswith("encoder.encoder.") 
        for k in state_dict.keys()
    )
    if needs_remap:
        new_state_dict = {}
        for k, v in state_dict.items():
            if k.startswith("encoder.") and not k.startswith("encoder.encoder."):
                new_state_dict["encoder." + k] = v
            else:
                new_state_dict[k] = v
        state_dict = new_state_dict
    
    # 5) ALLE Gewichte (Encoder + Decoder + Heads) aus dem Checkpoint laden
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()
    return model, detected_size


# ==================== INFERENCE ====================

def reconstruct_predictions(preds_by_slice, total_spec_columns, ID_TO_CLUSTER):
    grouped_preds = defaultdict(list)
    for ps in preds_by_slice:
        grouped_preds[ps["original_idx"]].append(ps)

    sec_per_col = 0.02
    cols_per_segment = total_spec_columns // 2
    all_preds_final = {"onset": [], "offset": [], "cluster": [], "score": [], "orig_idx": []}

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
                    all_preds_final["cluster"].append(ID_TO_CLUSTER.get(c, "unknown"))
                    all_preds_final["score"].append(float(score))
                    all_preds_final["orig_idx"].append(orig_idx)

    return all_preds_final



# ==================== CONSOLIDATION ====================

def consolidate_preds(all_preds_runs, overlap_tolerance=0.1):
    consolidated = {"onset": [], "offset": [], "cluster": [], "score": [], "orig_idx": []}
    combined_preds = []

    for run_idx, preds in enumerate(all_preds_runs):
        for o, f, c, s, oi in zip(preds["onset"], preds["offset"], preds["cluster"], preds["score"], preds["orig_idx"]):
            combined_preds.append({"onset": o, "offset": f, "cluster": c, "score": s, "orig_idx": oi, "run": run_idx})

    for cluster in set(p["cluster"] for p in combined_preds):
        cluster_preds = [p for p in combined_preds if p["cluster"] == cluster]
        used = set()

        for i, p1 in enumerate(cluster_preds):
            if i in used:
                continue
            overlapping_runs = [p1["run"]]
            overlap_candidates = [p1]

            for j, p2 in enumerate(cluster_preds):
                if i == j or j in used:
                    continue
                intersection = max(0, min(p1["offset"], p2["offset"]) - max(p1["onset"], p2["onset"]))
                union = max(p1["offset"], p2["offset"]) - min(p1["onset"], p2["onset"])
                iou = intersection / union if union > 0 else 0
                if iou > overlap_tolerance:
                    overlapping_runs.append(p2["run"])
                    overlap_candidates.append(p2)
                    used.add(j)

            if len(overlapping_runs) >= 2:
                best_pred = max(overlap_candidates, key=lambda x: x["score"])
                consolidated["onset"].append(best_pred["onset"])
                consolidated["offset"].append(best_pred["offset"])
                consolidated["cluster"].append(best_pred["cluster"])
                consolidated["score"].append(best_pred["score"])

    return consolidated


def consolidate_preds_per_file(all_preds_runs, overlap_tolerance=0.1):
    from collections import defaultdict

    consolidated = {"onset": [], "offset": [], "cluster": [], "score": [], "orig_idx": []}

    # Alle Predictions nach orig_idx gruppieren
    preds_by_file = defaultdict(list)
    for run_idx, preds in enumerate(all_preds_runs):
        for o, f, c, s, oi in zip(preds["onset"], preds["offset"], preds["cluster"], preds["score"], preds["orig_idx"]):
            preds_by_file[oi].append({"onset": o, "offset": f, "cluster": c, "score": s, "run": run_idx})

    # Für jede Datei separat konsolidieren
    for orig_idx, file_preds in preds_by_file.items():
        if len(file_preds) == 0:
            continue  # keine Predictions für diese Datei

        for cluster in set(p["cluster"] for p in file_preds):
            cluster_preds = [p for p in file_preds if p["cluster"] == cluster]
            used = set()

            for i, p1 in enumerate(cluster_preds):
                if i in used:
                    continue

                overlapping_runs = [p1["run"]]
                overlap_candidates = [p1]

                for j, p2 in enumerate(cluster_preds):
                    if i == j or j in used:
                        continue
                    intersection = max(0, min(p1["offset"], p2["offset"]) - max(p1["onset"], p2["onset"]))
                    union = max(p1["offset"], p2["offset"]) - min(p1["onset"], p2["onset"])
                    iou = intersection / union if union > 0 else 0
                    if iou > overlap_tolerance:
                        overlapping_runs.append(p2["run"])
                        overlap_candidates.append(p2)
                        used.add(j)

                # Nur wenn es mindestens 1 Kandidat gibt
                if len(overlap_candidates) > 0 and len(overlapping_runs) >= 2:
                    best_pred = max(overlap_candidates, key=lambda x: x["score"])
                    consolidated["onset"].append(best_pred["onset"])
                    consolidated["offset"].append(best_pred["offset"])
                    consolidated["cluster"].append(best_pred["cluster"])
                    consolidated["score"].append(best_pred["score"])
                    consolidated["orig_idx"].append(orig_idx)

    return consolidated


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
    parser.add_argument("--threshold", type=float, default=0)
    parser.add_argument("--iou_threshold", type=float, default=0.4)
    parser.add_argument("--overlap_tolerance", type=float, default=0.1)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--num_decoder_layers", type=int, default=3)
    parser.add_argument("--num_head_layers", type=int, default=2)
    parser.add_argument("--low_quality_value", type=float, default=0.5)
    parser.add_argument("--value_q2", type=float, default=1)
    parser.add_argument("--allowed_qualities", default=[1,2])
    parser.add_argument("--num_workers", type=int, default=1)
    parser.add_argument("--centerframe_size", type=float, default=0.6)
    parser.add_argument("--whisper_size", type=str, default=None, 
                        help="Whisper model size: 'base' or 'large'. If not specified, will be auto-detected from checkpoint path.")

    args = parser.parse_args()

    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    save_dir = os.path.join(args.output_dir, timestamp)
    os.makedirs(save_dir, exist_ok=True)

    with open(os.path.join(save_dir, "run_arguments.json"), "w") as f:
        json.dump(vars(args), f, indent=2)


    audio_paths, label_paths = get_audio_and_label_paths_from_folders(args.audio_folder, args.label_folder)
    cluster_codebook = FIXED_CLUSTER_CODEBOOK
    id_to_cluster = ID_TO_CLUSTER

    # Modell laden und Whisper-Größe erkennen
    model, detected_whisper_size = load_trained_whisperformer(
        args.checkpoint_path, 
        args.num_classes, 
        args.num_decoder_layers, 
        args.num_head_layers, 
        args.device,
        whisper_size=args.whisper_size
    )
    
    # Feature Extractor laden (identisch bei allen Whisper-Groessen: 80 Mel-Bins, 16kHz)
    _project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
    _whisper_models_dir = os.path.join(_project_root, "whisper_models")
    feature_extractor = WhisperFeatureExtractor.from_pretrained(
        os.path.join(_whisper_models_dir, "whisper_base"),
        local_files_only=True
    )


    all_preds_final  = {"onset": [], "offset": [], "cluster": [], "score": [], "orig_idx": []}
    for audio_path, label_path in zip(audio_paths, label_paths):
        print(f"\n===== Processing {os.path.basename(audio_path)} =====")
        audio_list, label_list = load_data([audio_path], [label_path], cluster_codebook=cluster_codebook, n_threads=1)
        # ======= Drei Slicing-Durchläufe mit Offsets =======
        shift_offsets = [0, 1000, 2000]
        all_preds_runs = []

        for offset_idx, offset in enumerate(shift_offsets):
            audio_list_shifted, label_list_shifted, metadata_list_shifted = slice_audios_and_labels(
                audio_list, label_list, args.total_spec_columns, offset=offset
            )
            print(f"[Run {offset_idx+1}] Created {len(audio_list_shifted)} slices with offset {offset}")

            val_dataset = WhisperFormerDatasetQuality(
                audio_list_shifted, label_list_shifted, args.total_spec_columns,
                feature_extractor, args.num_classes, args.low_quality_value, args.value_q2, args.centerframe_size
            )
            val_dataloader = DataLoader(
                val_dataset, batch_size=args.batch_size, shuffle=False,
                num_workers=args.num_workers, collate_fn=collate_fn, drop_last=False
            )

            preds_by_slice = run_inference_new(
                model=model,
                dataloader=val_dataloader,
                device=args.device,
                threshold=args.threshold,
                iou_threshold=args.iou_threshold,
                metadata_list=metadata_list_shifted
            )

            final_preds_shifted = reconstruct_predictions(preds_by_slice, args.total_spec_columns, ID_TO_CLUSTER)



            all_preds_runs.append(final_preds_shifted)

        # ======= Konsolidierung für jedes File einzeln=======

        final_preds = consolidate_preds(all_preds_runs, overlap_tolerance=args.overlap_tolerance)

        # Predictions pro Datei speichern
        base_name = os.path.splitext(os.path.basename(audio_path))[0]
        json_path = os.path.join(save_dir, f"{base_name}_preds.json")
        with open(json_path, "w") as f:
            json.dump(final_preds, f, indent=2)
        print(f"✅ Predictions saved to {json_path}")

    