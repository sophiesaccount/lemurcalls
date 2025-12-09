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
from whisperformer_model_large import WhisperFormer
from transformers import WhisperModel, WhisperFeatureExtractor
from datautils import (
    get_audio_and_label_paths_from_folders,
    load_data,
    slice_audios_and_labels,
    FIXED_CLUSTER_CODEBOOK,
    ID_TO_CLUSTER
)
from whisperformer_train import collate_fn, nms_1d_torch, evaluate_detection_metrics_with_false_class_qualities
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from sklearn.metrics import precision_score, recall_score, f1_score
from scatterplot_ampl_snr_score import compute_snr, compute_snr_new, compute_snr_timebased

import matplotlib.pyplot as plt
import librosa.display

import matplotlib.pyplot as plt
import librosa.display
import numpy as np
import os
import matplotlib.patches as mpatches
import glob


def plot_spectrogram_and_scores(
    mel_spec,
    class_scores,
    gt_onsets,
    gt_offsets,
    gt_classes,
    gt_qualities,
    pred_onsets,
    pred_offsets,
    pred_classes,
    segment_idx,
    save_dir,
    base_name,
    y,
    threshold=0.35,
    ID_TO_CLUSTER=None,
    extra_onsets=None,
    extra_offsets=None,
    extra_labels=None,
    extra_title="WhisperSeg Predictions",
    gt_title="GT Labels with Quality Classes"
):
    """
    Plottet Mel-Spektrogramm + Scores + Ground Truth + zusätzliche Labels.

    Args:
      mel_spec: (num_mels, T)
      class_scores: (T, num_classes)
      gt_onsets/gt_offsets: Sekundenangaben der Labels im Segment
      gt_classes: Klassen der Labels
      gt_qualities: Qualitätsklassen (1, 2, 3, ...)
      extra_onsets/extra_offsets/extra_labels: optionale Zusatzlabels (Listen)
    """
    import matplotlib.pyplot as plt
    import librosa.display
    import numpy as np
    import os

    T = class_scores.shape[0]
    num_classes = class_scores.shape[1]
    sec_per_col = 0.02  # WhisperFeatureExtractor → 50 Hz
    time_axis = np.arange(T) * sec_per_col

    # Farben
    colors = plt.cm.Set2(np.linspace(0, 1, num_classes))
    color_map = {i: colors[i % len(colors)] for i in range(num_classes)}

    color_map = {0: 'darkorange', 1: 'cornflowerblue', 2: 'gold', 3: 'r'}
    map = {'m': 'moan','h': 'hmm', 'w': 'wail'}
    unique_labels = ["m", "h", "w"]

    # Wenn Extra-Labels existieren → 4 Reihen, sonst 3
    has_extra = extra_onsets is not None and extra_offsets is not None and extra_labels is not None
    nrows = 4 if has_extra else 3
    height_ratios = [3, 1, 0.6] + ([0.6] if has_extra else [])

    fig, axs = plt.subplots(nrows, 1, figsize=(12, 7 if has_extra else 6), height_ratios=height_ratios)
    if nrows == 3:
        ax_spec, ax_scores, ax_gt = axs
    else:
        ax_spec, ax_scores, ax_gt, ax_extra = axs

    # 1️⃣ Mel-Spektrogramm
    # mel_spec: (n_mels, T)
    n_mels = mel_spec.shape[0]
    fmin = 0
    fmax = 8000

    # Frequenzen der Mel-Bins in Hz
    mel_f = librosa.mel_frequencies(n_mels=n_mels, fmin=fmin, fmax=fmax)

        # 1️⃣ Mel-Spektrogramm plotten
    librosa.display.specshow(
        mel_spec,
        cmap=plt.cm.magma,
        sr=16000,
        hop_length=160,
        x_axis='time',
        y_axis='mel',  # <-- sorgt für die richtige Mel-Skala
        fmin=0,
        fmax=8000,
        ax=ax_spec
    )
    ax_spec.set_ylabel("Frequency (Hz)")

    # 2️⃣ Scores + Ground Truth
    frame_width = sec_per_col
    for c in range(num_classes):
        ax_scores.bar(
            time_axis,
            class_scores[:, c],
            width=frame_width,
            align='edge',
            alpha=1,
            label=f"{map[ID_TO_CLUSTER[c]]}",
            color=color_map[c]
        )

    ax_scores.axhline(y=threshold, color='r', linestyle='--', label=f"Threshold {threshold}")
    ax_scores.set_ylim(0, 1.1)
    ax_scores.set_title("WhisperFormer Scores and Labels")
    ax_scores.set_xlabel("Time (s)")
    ax_scores.set_ylabel("Score")
    ax_scores.set_xlim(0, T * sec_per_col)

    # 🎯 WhisperFormer Predictions
    for onset, offset, c in zip(pred_onsets, pred_offsets, pred_classes):
        color = color_map[FIXED_CLUSTER_CODEBOOK[c]]
        ax_scores.axvspan(onset, offset, color=color, alpha=0.3)
        mid = (onset + offset) / 2

        #calculate SNR
        sr=16000
        start_sample = int(onset * sr)
        end_sample = int(offset * sr)
        segment_audio = y
        snr_value = compute_snr_new(segment_audio, sr, cutoff=200)
        snr_value = compute_snr_timebased(segment_audio, sr, start_sample, end_sample)

    patches = [mpatches.Patch(color=color_map[FIXED_CLUSTER_CODEBOOK[l]], label=str(map[l])) for l in unique_labels]
    ax_scores.legend(handles=patches, loc="upper right")


    # 🔁 Legende bereinigen
    handles, labels = ax_scores.get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    ax_scores.legend(by_label.values(), by_label.keys(), loc="upper right")

    # 2 Ground Truth Labels 
    if has_extra:
        ax_gt.set_title(gt_title)
        ax_gt.set_xlim(0, T * sec_per_col)
        ax_gt.set_ylim(0, 1)
        ax_gt.set_xlabel("Time (s)")
        ax_gt.set_yticks([])

        
        #unique_labels = sorted(set(gt_classes))
        unique_labels = ["m", "h", "w"]
        label_colors = plt.cm.Paired(np.linspace(0, 1, len(unique_labels)))
        label_color_map = {lab: label_colors[i] for i, lab in enumerate(unique_labels)}
        #farben_extra = farben[np.array([np.where(np.unique(gt_labels) == lab)[0][0] for lab in extra_labels])]

        for onset, offset, lab, q in zip(gt_onsets, gt_offsets, gt_classes, gt_qualities):
            #color = label_color_map[lab]
            color = color_map[FIXED_CLUSTER_CODEBOOK[lab]]
            ax_gt.axvspan(onset, offset, color=color, alpha=0.4)
            mid = (onset + offset) / 2
            ax_gt.text(
                mid, 0.5, str(q),
                ha='center',
                va='center',
                fontsize=8,
                fontweight='bold',
                color='black',
                bbox=dict(facecolor='white', edgecolor='none', alpha=0.7, pad=0.5)
            )

        # Legende für GT Labelscolor = color_map[FIXED_CLUSTER_CODEBOOK[c]]
        patches = [mpatches.Patch(color=color_map[FIXED_CLUSTER_CODEBOOK[l]], label=str(map[l])) for l in unique_labels]
        ax_gt.legend(handles=patches, loc="upper right")

    
    # 3️⃣ Zusätzliche Labels (optional)
    if has_extra:
        ax_extra.set_title(extra_title)
        ax_extra.set_xlim(0, T * sec_per_col)
        ax_extra.set_ylim(0, 1)
        ax_extra.set_xlabel("Time (s)")
        ax_extra.set_yticks([])

        #unique_labels = sorted(set(extra_labels))
        unique_labels = ["m", "h", "w"]
        label_colors = plt.cm.Paired(np.linspace(0, 1, len(unique_labels)))
        label_color_map = {lab: label_colors[i] for i, lab in enumerate(unique_labels)}
        #farben_extra = farben[np.array([np.where(np.unique(gt_labels) == lab)[0][0] for lab in extra_labels])]

        for onset, offset, lab in zip(extra_onsets, extra_offsets, extra_labels):
            #color = label_color_map[lab]
            color = color_map[FIXED_CLUSTER_CODEBOOK[lab]]
            ax_extra.axvspan(onset, offset, color=color, alpha=0.4)
            mid = (onset + offset) 
            """
            ax_extra.text(
                mid, 0.5, str(lab),
                ha='center',
                va='center',
                fontsize=8,
                fontweight='bold',
                color='black',
                bbox=dict(facecolor='white', edgecolor='none', alpha=0.7, pad=0.5)
            )
            """

        # Legende für Extra Labels
        patches = [mpatches.Patch(color=color_map[FIXED_CLUSTER_CODEBOOK[l]], label=str(map[l])) for l in unique_labels]
        ax_extra.legend(handles=patches, loc="upper right")

    plt.tight_layout()

    save_filename = f"{base_name}_segment_{segment_idx:02d}_spectrogram_scores_gt.png"
    save_path = os.path.join(save_dir, save_filename)
    plt.savefig(save_path, dpi=150)
    plt.close()
    #print(f"✅ Segment-Plot gespeichert unter {save_path}")

# ==================== MODEL LOADING ====================

def load_trained_whisperformer(checkpoint_path, num_classes, num_decoder_layers, num_head_layers, device):
    #whisper_model = WhisperModel.from_pretrained("openai/whisper-small", local_files_only=True)
    whisper_model = WhisperModel.from_pretrained("/projects/extern/CIDAS/cidas_digitalisierung_lehre/mthesis_sophie_dierks/dir.project/lemurcalls/lemurcalls/whisper_models/whisper_large")

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
    parser.add_argument("--pred_label_folder", required=True)
    parser.add_argument("--extra_label_folder", type=str, default=None)
    parser.add_argument("--output_dir", default="inference_outputs")
    parser.add_argument("--total_spec_columns", type=int, default=3000)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--num_classes", type=int, default=1)
    parser.add_argument("--threshold", type=float, default=0.35)
    parser.add_argument("--iou_threshold", type=float, default=0.4)
    parser.add_argument("--overlap_tolerance", type=float, default=0.1)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--num_decoder_layers", type = int, default = 3)
    parser.add_argument("--num_head_layers", type = int, default = 2)
    parser.add_argument("--low_quality_value", type=float, default=0.5)
    parser.add_argument("--value_q2", type=float, default=1)
    parser.add_argument("--centerframe_size", type=float, default=0.6)
    parser.add_argument("--allowed_qualities", default = [1,2,3])
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
    #feature_extractor = WhisperFeatureExtractor.from_pretrained("openai/whisper-small", local_files_only=True)

    feature_extractor = WhisperFeatureExtractor.from_pretrained("/projects/extern/CIDAS/cidas_digitalisierung_lehre/mthesis_sophie_dierks/dir.project/lemurcalls/lemurcalls/whisper_models/whisper_large",local_files_only=True)

    all_labels = {"onset": [], "offset": [], "cluster": [], "quality": []}
    all_preds_final  = {"onset": [], "offset": [], "cluster": [], "score": []}

    for audio_path, label_path in zip(audio_paths, label_paths):
        #print(f"\n===== Processing {os.path.basename(audio_path)} =====")
        audio_list, label_list = load_data([audio_path], [label_path], cluster_codebook=cluster_codebook, n_threads=1)
        audio_list, label_list, metadata_list = slice_audios_and_labels(audio_list, label_list, args.total_spec_columns)

        dataset = WhisperFormerDatasetQuality(audio_list, label_list, args.total_spec_columns, feature_extractor, args.num_classes, args.low_quality_value, args.value_q2, args.centerframe_size)
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
        # Audio laden
        y, sr = librosa.load(audio_path, sr=16000)

        # Labels laden
        with open(label_path, "r") as f:
            labels = json.load(f)
        
        clusters = labels["cluster"]
        labels["cluster"] = [ID_TO_CLUSTER[FIXED_CLUSTER_CODEBOOK[c]] for c in clusters]

        # Quality-Klassen hinzufügen
        if "quality" in labels:
            quality_list = labels["quality"]
        else:
            quality_list = ["unknown"] * len(labels["onset"])

        # === WhisperFormer Prediction Labels für diese Audio-Datei laden ===
        pred_labels_data = None
        if args.pred_label_folder is not None:
            base_name = os.path.splitext(os.path.basename(audio_path))[0]
            pattern_jsonr = os.path.join(args.pred_label_folder, f"{base_name}*.jsonr")
            pattern_json = os.path.join(args.pred_label_folder, f"{base_name}*.json")
            matching_files = sorted(glob.glob(pattern_json) + glob.glob(pattern_jsonr))

            if len(matching_files) > 0:
                pred_path = matching_files[0]  # Nimm die erste gefundene Datei
                with open(pred_path, "r") as f:
                    pred_labels_data = json.load(f)
                print(f"📘 WhisperFormer Predictions für {base_name} geladen aus {pred_path}")
            else:
                print(f"ℹ️ Keine passenden WhisperFormer Predictions gefunden für {base_name}")

        # === Extra Labels für diese Audio-Datei laden (optional) ===
        extra_labels_data = None
        if args.extra_label_folder is not None:
            base_name = os.path.splitext(os.path.basename(audio_path))[0]
            pattern_jsonr = os.path.join(args.extra_label_folder, f"{base_name}*.jsonr")
            pattern_json = os.path.join(args.extra_label_folder, f"{base_name}*.json")
            matching_files = sorted(glob.glob(pattern_json) + glob.glob(pattern_jsonr))

            if len(matching_files) > 0:
                extra_path = matching_files[0]  # Nimm die erste gefundene Datei
                with open(extra_path, "r") as f:
                    extra_labels_data = json.load(f)
                print(f"📘 Extra Labels für {base_name} geladen aus {extra_path}")
            else:
                print(f"ℹ️ Keine passenden Extra Labels gefunden für {base_name}")


        # === Visualisierung der ersten 3 Segmente mit Ground Truth ===
        if len(dataset) > 0:
            base_name = os.path.splitext(os.path.basename(audio_path))[0]
            #print(f"🔍 Visualisiere erste 3 Segmente von {base_name} mit Ground Truth ...")

            # Labels für die gesamte Datei laden
            gt_onsets = np.array(labels["onset"])
            gt_offsets = np.array(labels["offset"])
            gt_classes = np.array(labels["cluster"])
            gt_qualities = np.array(labels["quality"])

            # Dauer pro Segment (Sekunden)
            seg_dur = (args.total_spec_columns / 2) * 0.02  # da Whisper T/2 Frames -> 20 ms/Frame
            for i in range(min(3, len(dataset))):
                seg_start = i * seg_dur
                seg_end = (i + 1) * seg_dur

                #y anpassen
                y_part = y[int(seg_start*sr) : int(seg_end*sr)]

                # Ground Truth Events im Segment auswählen
                in_seg = (gt_onsets < seg_end) & (gt_offsets > seg_start)
                gt_onsets_seg = gt_onsets[in_seg] - seg_start
                gt_offsets_seg = gt_offsets[in_seg] - seg_start
                gt_classes_seg = gt_classes[in_seg]
                gt_qualities_seg = gt_qualities[in_seg]

                if not pred_labels_data:
                    pred_onsets = np.array([])
                    pred_offsets = np.array([])
                    pred_labels = np.array([])
                    pred_scores = np.array([])
                else:
                    pred_onsets  = np.array(pred_labels_data.get("onset", []))
                    pred_offsets = np.array(pred_labels_data.get("offset", []))
                    pred_labels  = np.array(pred_labels_data.get("cluster", []))
                    pred_scores  = np.array(pred_labels_data.get("score", []))

                # Segmentfilter
                in_seg = (pred_onsets < seg_end) & (pred_offsets > seg_start)

                # Score-Filter
                score_mask = pred_scores > args.threshold

                # kombinierter Filter
                mask = in_seg & score_mask

                pred_onsets_seg = pred_onsets[mask] - seg_start
                pred_offsets_seg = pred_offsets[mask] - seg_start
                pred_labels_seg = pred_labels[mask]
                pred_scores_seg = pred_scores[mask]
                """
                #etra labels im segment auswählen
                if extra_labels_data is not None:
                    extra_onsets = np.array(extra_labels_data.get("onset", []))
                    extra_offsets = np.array(extra_labels_data.get("offset", []))
                    extra_labels = np.array(extra_labels_data.get("cluster", []))

                    in_seg = (extra_onsets < seg_end) & (extra_offsets > seg_start)
                    extra_onsets_seg = extra_onsets[in_seg] - seg_start
                    extra_offsets_seg = extra_offsets[in_seg] - seg_start
                    extra_labels_seg = extra_labels[in_seg]
                """

                if not extra_labels_data:
                    extra_onsets = np.array([])
                    extra_offsets = np.array([])
                    extra_labels = np.array([])
                    #pred_scores = np.array([])
                else:
                    extra_onsets  = np.array(extra_labels_data.get("onset", []))
                    extra_offsets = np.array(extra_labels_data.get("offset", []))
                    extra_labels  = np.array(extra_labels_data.get("cluster", []))
                    #_scores  = np.array(extra_labels_data.get("score", []))
                
                in_seg = (extra_onsets < seg_end) & (extra_offsets > seg_start)
                extra_onsets_seg = extra_onsets[in_seg] - seg_start
                extra_offsets_seg = extra_offsets[in_seg] - seg_start
                extra_labels_seg = extra_labels[in_seg]
                  
                    

                # Hole Input-Feature (Mel)
                features = dataset[i]["input_features"].squeeze(0).cpu().numpy()  # (80, 3000)
                mel_spec = features

                # Modell-Scores
                with torch.no_grad():
                    x = dataset[i]["input_features"].unsqueeze(0).to(args.device)
                    class_preds, _ = model(x)
                    class_scores = torch.sigmoid(class_preds).squeeze(0).cpu().numpy()                


                plot_spectrogram_and_scores(
                    mel_spec=mel_spec,
                    class_scores=class_scores,
                    gt_onsets=gt_onsets_seg,
                    gt_offsets=gt_offsets_seg,
                    gt_classes=gt_classes_seg,
                    gt_qualities=gt_qualities_seg,
                    pred_onsets=pred_onsets_seg,
                    pred_offsets=pred_offsets_seg,
                    pred_classes=pred_labels_seg,
                    segment_idx=i,
                    save_dir=save_dir,
                    base_name=base_name,
                    y=y_part,
                    threshold=args.threshold,
                    ID_TO_CLUSTER=ID_TO_CLUSTER,
                    extra_onsets=extra_onsets_seg,
                    extra_offsets=extra_offsets_seg,
                    extra_labels=extra_labels_seg,
                )

