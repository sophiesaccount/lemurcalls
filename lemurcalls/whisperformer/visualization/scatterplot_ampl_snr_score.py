import argparse
import json
import os
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from datetime import datetime
from collections import defaultdict
import numpy as np
import librosa
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

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
from ..train import collate_fn, nms_1d_torch, evaluate_detection_metrics_with_false_class_qualities
from ..infer import soft_nms_1d_torch
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from sklearn.metrics import precision_score, recall_score, f1_score

import matplotlib.pyplot as plt
import librosa.display

import matplotlib.pyplot as plt
import librosa.display
import numpy as np
import os
import matplotlib.patches as mpatches

from scipy.signal import butter, filtfilt

ID_TO_CLUSTER = {
    0: "m",
    1: "t",     # representative name for trill/lt/h
    2: "w"
}

def highpass_filter(y, sr, cutoff=200, order=5):
    """
    Wendet einen Butterworth-Highpass-Filter an, um tiefe Frequenzen unterhalb von 'cutoff' Hz zu entfernen.
    """
    nyquist = 0.5 * sr
    normal_cutoff = cutoff / nyquist
    b, a = butter(order, normal_cutoff, btype='high', analog=False)
    y_filtered = filtfilt(b, a, y)
    return y_filtered

def compute_snr(y, sr, cutoff=200, order=5):
    """
    Berechnet eine einfache Signal-to-Noise-Ratio (SNR) in dB.
    'Signal' = Energie oberhalb cutoff Hz
    'Noise'  = Energie unterhalb cutoff Hz
    """
    # Highpass -> Signalanteil über cutoff
    y_signal = highpass_filter(y, sr, cutoff=cutoff, order=order)
    # Lowpass -> Restenergie unter cutoff
    nyquist = 0.5 * sr
    normal_cutoff = cutoff / nyquist
    b, a = butter(order, normal_cutoff, btype='low', analog=False)
    y_noise = filtfilt(b, a, y)

    # Energie (mittlere Leistung)
    signal_power = np.mean(y_signal ** 2) + 1e-10
    noise_power = np.mean(y_noise ** 2) + 1e-10

    snr_db = 10 * np.log10(signal_power / noise_power)
    return snr_db

    import numpy as np

def compute_snr_timebased(y, sr, start_sample, end_sample):
    """
    Berechnet SNR in dB anhand eines Audiosegments und der direkt folgenden "Noise"-Region.
    
    Args:
        y: Audio-Signal (1D numpy array)
        sr: Samplingrate
        start_sample: Start des Signal-Segments
        end_sample: Ende des Signal-Segments
    
    Returns:
        snr_db: SNR in dB
    """
    # Signalsegment
    y_signal = y[start_sample:end_sample]

    # Noise-Segment direkt danach (gleich lang)
    noise_start = end_sample
    noise_end = end_sample + (end_sample - start_sample)
    
    # Prüfen, dass wir nicht über das Audio hinauslaufen
    #if noise_end > len(y):
    #    noise_end = len(y)
    y_noise = y[noise_start:noise_end]
    #print(f'noise_start: {noise_end}')
    #print(f'noise_end: {noise_end}')

    # Falls kein Noise-Segment verfügbar ist, kleine Zahl hinzufügen, um Division durch 0 zu vermeiden
    #if len(y_noise) == 0:
    #if noise_end >=16000:
    #    y_noise = np.array([1e-10])
    #    print('No Noise Segment found!')

    # Energie (mittlere Leistung)
    signal_power = np.mean(y_signal ** 2) + 1e-10
    noise_power = np.mean(y_noise ** 2) + 1e-10
    #print(f'signal_power: {signal_power}')
    #print(f'noise_power: {signal_power}')

    snr_db = 10 * np.log10(signal_power / noise_power)
    #print(f'snr_db: {snr_db}')
    return snr_db



from scipy.signal import butter, filtfilt
import numpy as np

def compute_snr_top200(y, sr, topband_hz=200, order=5):
    """
    Berechnet eine einfache Signal-to-Noise-Ratio (SNR) in dB.
    'Noise'  = Energie in den obersten `topband_hz` Hz (High-Frequency-Anteil)
    'Signal' = Restenergie unterhalb Nyquist - topband_hz
    """
    nyquist = 0.5 * sr
    cutoff = nyquist - topband_hz  # Grenze zwischen Signal und Noise
    if cutoff <= 0:
        raise ValueError("Samplingrate zu niedrig oder topband_hz zu groß!")

    # === Lowpass: Signalanteil unterhalb cutoff ===
    normal_cutoff = cutoff / nyquist
    b_low, a_low = butter(order, normal_cutoff, btype='low', analog=False)
    y_signal = filtfilt(b_low, a_low, y)

    # === Highpass: Noise-Anteil oberhalb cutoff ===
    b_high, a_high = butter(order, normal_cutoff, btype='high', analog=False)
    y_noise = filtfilt(b_high, a_high, y)

    # === Leistung (Energie) ===
    signal_power = np.mean(y_signal ** 2) + 1e-10
    noise_power = np.mean(y_noise ** 2) + 1e-10

    snr_db = 10 * np.log10(signal_power / noise_power)
    return snr_db

from scipy.signal import butter, filtfilt

def compute_snr_new(y, sr, cutoff=200, signal_high=1200, order=5):
    """
    Berechnet die SNR in dB.
    Signal = Energie zwischen cutoff und signal_high Hz
    Noise  = Energie unterhalb cutoff Hz
    """
    nyquist = 0.5 * sr
    
    # 1️⃣ Noise: Tiefpass unter cutoff
    normal_cutoff = cutoff / nyquist
    b, a = butter(order, normal_cutoff, btype='low')
    y_noise = filtfilt(b, a, y)
    
    """
    # 1️⃣ Noise: Tiefpass unter cutoff
    low = 100
    high = cutoff 
    b, a = butter(order, [low/nyquist, high/nyquist], btype='band')

    y_noise = filtfilt(b, a, y)
    """
    

    # 2️⃣ Signal: Bandpass cutoff - signal_high
    high = min(signal_high, nyquist)  # sicherstellen, dass wir nicht über Nyquist gehen
    low  = cutoff
    b, a = butter(order, [low/nyquist, high/nyquist], btype='band')
    y_signal = filtfilt(b, a, y)

    # Energie (mittlere Leistung)
    signal_power = np.mean(y_signal**2) + 1e-10
    noise_power  = np.mean(y_noise**2) + 1e-10

    snr_db = 10 * np.log10(signal_power / noise_power)
    return snr_db



def plot_spectrogram_and_scores(
    mel_spec,
    class_scores,
    gt_onsets,
    gt_offsets,
    gt_classes,
    segment_idx,
    save_dir,
    base_name,
    threshold=0.35,
    ID_TO_CLUSTER=None
):
    """
    Plottet Mel-Spektrogramm + Scores + Ground Truth für ein Segment.
    Args:
      mel_spec: (num_mels, T)
      class_scores: (T, num_classes)
      gt_onsets/gt_offsets: Listen mit Sekundenangaben der Labels im Segment
      gt_classes: Klassen der Labels
      segment_idx: Segment-Index
      base_name: Dateiname (ohne Endung)
      ID_TO_CLUSTER: optionales Mapping Klasse -> Name
    """
    T = class_scores.shape[0]
    num_classes = class_scores.shape[1]
    #sec_per_col = 0.02  # WhisperFeatureExtractor → 50 Hz, also 20 ms / Frame
    downsample_ratio = mel_spec.shape[1] / class_scores.shape[0]
    sec_per_col = 0.02 * downsample_ratio
    time_axis = np.arange(T) * sec_per_col

    color_map = {0: 'darkorange', 1: 'cornflowerblue', 2: 'gold', 3: 'r'}

    fig, axs = plt.subplots(2, 1, figsize=(12, 6), height_ratios=[3, 1])

    # 1️⃣ Mel-Spektrogramm
    librosa.display.specshow(mel_spec, sr=16000, hop_length=320, x_axis="time", y_axis="mel", ax=axs[0])
    axs[0].set_title(f"{base_name} – Segment {segment_idx}: Mel-Spektrogramm")
    axs[0].set_xlabel("")
    axs[0].set_ylabel("Mel-Frequenz (Hz)")

    # 2️⃣ Scores
    frame_width = sec_per_col 
    for c in range(num_classes):
        #axs[1].plot(time_axis, class_scores[:, c], label=f"Class {ID_TO_CLUSTER[c]}", alpha=0.8)
        axs[1].bar(time_axis, class_scores[:, c], width=frame_width, align='edge', alpha=0.6, label=f"Class {ID_TO_CLUSTER[c]}",color=color_map[FIXED_CLUSTER_CODEBOOK[ID_TO_CLUSTER[c]]])

    axs[1].axhline(y=threshold, color='r', linestyle='--', label=f"Threshold {threshold}")
    axs[1].set_ylim(0, 1)
    axs[1].set_title("Scores + Ground Truth")
    axs[1].set_xlabel("Zeit (s)")
    axs[1].set_ylabel("Score")
    axs[1].set_xlim(0, T * sec_per_col) 
    
    """
    # 🎯 Ground Truth Overlay
    colors = plt.cm.tab10(np.linspace(0, 1, num_classes))
   
    for onset, offset, c in zip(gt_onsets, gt_offsets, gt_classes):
        axs[1].axvspan( onset*2, offset*2, color=colors[c % len(colors)], alpha=0.3, label=f"GT class {ID_TO_CLUSTER[c]}")

            colors = plt.cm.tab10(np.linspace(0, 1, num_classes))  
    for c in range(num_classes):
        mask = np.array(labels_cluster) == c
        for onset, offset in zip(labels_onset_sec[mask], labels_offset_sec[mask]):
            axs[1].axvspan(onset, offset, color=colors[c % len(colors)], alpha=0.3, label=f"GT {ID_TO_CLUSTER[c]}")
    """

    # Zeichnen
    for onset, offset, c in zip(gt_onsets, gt_offsets, gt_classes):
        axs[1].axvspan(
            2*onset, 2*offset, 
            color=color_map[FIXED_CLUSTER_CODEBOOK[c]],  # Mapping statt modulo
            alpha=0.3, 
            label=f"GT class {c}"
        )


    # 🔁 Legende bereinigen (duplikate entfernen)
    handles, labels = axs[1].get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    axs[1].legend(by_label.values(), by_label.keys(), loc="upper right")

    plt.tight_layout()

    save_filename = f"{base_name}_segment_{segment_idx:02d}_spectrogram_scores_gt.png"
    save_path = os.path.join(save_dir, save_filename)
    plt.savefig(save_path, dpi=150)
    plt.close()
    #print(f"✅ Segment-Plot gespeichert unter {save_path}")


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
            for k, v in batch.items():
                if isinstance(v, torch.Tensor):
                    batch[k] = v.to(device, non_blocking=True)

            use_autocast = (isinstance(device, str) and device.startswith("cuda")) or (hasattr(device, "type") and device.type == "cuda")
            autocast_ctx = torch.amp.autocast(device_type="cuda", dtype=torch.float16) if use_autocast else contextlib.nullcontext()

            with autocast_ctx:
                class_preds, regr_preds = model(batch["input_features"])
                class_probs = torch.sigmoid(class_preds)

            B, T, C = class_preds.shape
            for b in range(B):
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
                        intervals = soft_nms_1d_torch(intervals, iou_threshold=iou_threshold)
                        #intervals = nms_1d_torch(intervals, iou_threshold=iou_threshold)
                        intervals = intervals.cpu().tolist()
                    else:
                        intervals = []

                    preds_per_class.append({"class": c, "intervals": intervals})

                preds_by_slice.append({
                    "original_idx": meta["original_idx"],
                    "segment_idx": meta["segment_idx"],
                    "preds": preds_per_class,
                    # NEU: Scores je Frame (T x C) für spätere Mid-Frame-Auswertung
                    "scores": class_probs[b].detach().cpu().numpy()
                })

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
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--num_classes", type=int, default=3)
    parser.add_argument("--threshold", type=float, default=0.35)
    parser.add_argument("--iou_threshold", type=float, default=0.4)
    parser.add_argument("--overlap_tolerance", type=float, default=0.1)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--num_decoder_layers", type = int, default = 3)
    parser.add_argument("--num_head_layers", type = int, default = 2)
    parser.add_argument("--low_quality_value", type=float, default=0.5)
    parser.add_argument("--value_q2", type=float, default=1)
    parser.add_argument("--cutoff", type=int, default=200)
    parser.add_argument("--centerframe_size", type=float, default=0.6)
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
    all_preds  = {"onset": [], "offset": [], "cluster": [], "score": [], "original_idx": []}

    amplitudes = []
    scores = []
    classes = []
    qualities = []
    snrs = []

    sec_per_col = 0.02  # Whisper-Feature: 50 Hz → 20 ms / Frame

    # Mapping von Cluster-Label zu ID (um Klassenscores zu holen)
    cluster_to_id = {v: k for k, v in ID_TO_CLUSTER.items()}

    for i, (audio_path, label_path) in enumerate(zip(audio_paths, label_paths)):
        base_name = os.path.splitext(os.path.basename(audio_path))[0]
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
        for p in preds_by_slice:
            p["original_idx"] = i
        #print(f'preds_by_slice: {preds_by_slice}')
        
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
        """
        # Rekonstruiere Vorhersagen für diese Datei mit segment_idx
        file_preds = reconstruct_predictions(preds_by_slice, args.total_spec_columns, ID_TO_CLUSTER)
        

        # Original-Index hinzufügen
        file_preds["original_idx"] = [i] * len(file_preds["onset"])
        #print(file_preds)
        #{'onset': [64.725], 'offset': [65.505], 'cluster': ['m'], 'score': [0.83251953125], 'original_idx': [20]}

        # In den globalen Container einfügen
        all_preds["onset"].extend(file_preds["onset"])
        all_preds["offset"].extend(file_preds["offset"])
        all_preds["cluster"].extend(file_preds["cluster"])
        all_preds["score"].extend(file_preds["score"])
        all_preds["original_idx"].extend(file_preds["original_idx"])
        """
        # --- Lade Audio ---
        y, sr = librosa.load(audio_path, sr=16000)

        # --- Lade Ground Truth ---
        with open(label_path, "r") as f:
            labels = json.load(f)

        gt_onsets = np.array(labels["onset"])
        gt_offsets = np.array(labels["offset"])
        gt_clusters = np.array(labels["cluster"])
        gt_qualities = np.array(labels["quality"])

        # --- Hole Modell-Scores aus den gespeicherten Segment-Vorhersagen ---
        # Finde alle Slices, die zu dieser Datei gehören

        file_slices = [p for p in preds_by_slice if p["original_idx"] == i]

        #print(f'file_slices: {file_slices}')
        score_list = [p["scores"] for p in sorted(file_slices, key=lambda x: x["segment_idx"]) if "scores" in p and len(p["scores"]) > 0]


        if len(score_list) > 0:
            all_scores = np.concatenate(score_list, axis=0)
        else:
            all_scores = np.array([])  # oder None, je nach gewünschtem Verhalten
            print("⚠️ Keine Scores gefunden – file_slices ist leer oder enthält keine gültigen 'scores'.")

        # --- Pro Ground Truth Event ---
        for onset, offset, cluster_label, cluster_quality in zip(gt_onsets, gt_offsets, gt_clusters, gt_qualities):
            # Center-Time in Sekunden
            center_sec = 0.5 * (onset + offset)
            frame_idx = int(center_sec / sec_per_col)

            if frame_idx >= all_scores.shape[0]:
                continue

            # Hole passende Klasse-ID
            if cluster_label not in cluster_to_id:
                continue
            class_id = cluster_to_id[cluster_label]

            model_score = all_scores[frame_idx, class_id]

            # Berechne maximale Amplitude im Audiosegment
            start_sample = int(onset * sr)
            end_sample = int(offset * sr)

            segment_audio = y[start_sample:end_sample]
            if len(segment_audio) == 0:
                continue

            # Nur Frequenzen oberhalb von 200 Hz berücksichtigen
            segment_audio_filtered = highpass_filter(segment_audio, sr, cutoff=args.cutoff)
            max_amp = float(np.max(np.abs(segment_audio_filtered)))
            #max_amp = float(np.max(np.abs(segment_audio)))

            amplitudes.append(max_amp)
            scores.append(float(model_score))
            classes.append(cluster_label)
            qualities.append(cluster_quality)

            #snr_value = compute_snr(segment_audio, sr, cutoff=args.cutoff)
            #snr_value = compute_snr_timebased(y, sr, start_sample, end_sample)
            snr_value = compute_snr_new(segment_audio, sr, cutoff=args.cutoff,signal_high=1000)
            #print(cluster_quality, snr_value)

            #snr_value = compute_snr_top200(segment_audio, sr)

            snrs.append(snr_value)


    """

    # === Scatterplot erstellen ===
    if len(amplitudes) > 0:
        plt.figure(figsize=(8, 6))
        sns.scatterplot(x=amplitudes, y=scores, hue=classes, palette="Set2", alpha=0.7, edgecolor="k", s=60)
        plt.xlabel("Maximale Amplitude des GT-Segments")
        plt.ylabel("Model Score (Center Frame)")
        plt.title("Amplitude vs. Model Score pro Ground Truth Event")
        plt.grid(True, linestyle="--", alpha=0.6)
        plt.tight_layout()

        scatter_path = os.path.join(save_dir, "scatter_amplitude_vs_score.png")
        plt.savefig(scatter_path, dpi=150)
        plt.close()

        corr = np.corrcoef(amplitudes, scores)[0, 1]
        print(f"✅ Scatterplot gespeichert unter: {scatter_path}")
        print(f"📈 Korrelation (Amplitude–Score): {corr:.3f}")
    else:
        print("⚠️ Keine gültigen Punkte für Scatterplot gefunden.")

# === Zweiten Scatterplot erstellen ===
    if len(qualities) > 0:
        plt.figure(figsize=(8, 6))
        sns.scatterplot(x=qualities, y=scores, hue=classes, palette="Set2", alpha=0.7, edgecolor="k", s=60)
        plt.xlabel("Quality of GT Segments")
        plt.ylabel("Model Score at Center Frame")
        plt.title("Quality vs. Model Score per Ground Truth Event")
        plt.grid(True, linestyle="--", alpha=0.6)
        plt.tight_layout()

        scatter_path = os.path.join(save_dir, "scatter_quality_vs_score.png")
        plt.savefig(scatter_path, dpi=150)
        plt.close()

        #corr = np.corrcoef(qualities, scores)[0, 1]
        print(f"✅ Scatterplot gespeichert unter: {scatter_path}")
        #print(f"📈 Korrelation (Quality–Score): {corr:.3f}")
    else:
        print("⚠️ Keine gültigen Punkte für Scatterplot gefunden.")

    # === Dritten Scatterplot: SNR vs Model Score ===
    if len(snrs) > 0:
        plt.figure(figsize=(8, 6))
        sns.scatterplot(x=snrs, y=scores, hue=classes, palette="Set2", alpha=0.7, edgecolor="k", s=60)
        plt.xlabel("Signal-to-Noise Ratio (dB)")
        plt.ylabel("Model Score at Center Frame")
        plt.title("SNR vs. Model Score per Ground Truth Event")
        plt.grid(True, linestyle="--", alpha=0.6)
        plt.tight_layout()

        scatter_path = os.path.join(save_dir, "scatter_snr_vs_score.png")
        plt.savefig(scatter_path, dpi=150)
        plt.close()

        #corr = np.corrcoef(snrs, scores)[0, 1]
        print(f"✅ Scatterplot gespeichert unter: {scatter_path}")
        #print(f"📈 Korrelation (SNR–Score): {corr:.3f}")
    else:
        print("⚠️ Keine gültigen Punkte für SNR-Scatterplot gefunden.")

    # === Scatterplot: SNR vs Quality ===
    if len(snrs) > 0 and len(qualities) > 0:
        plt.figure(figsize=(8, 6))
        sns.boxplot(x=qualities, y=snrs, palette="Set2", showfliers=False)
        sns.stripplot(x=qualities, y=snrs, color="k", alpha=0.5, jitter=0.2)
        plt.xlabel("Quality Class")
        plt.ylabel("Signal-to-Noise Ratio (dB)")
        plt.title("SNR vs. Quality Class")
        plt.grid(True, linestyle="--", alpha=0.6)
        plt.tight_layout()

        snr_quality_path = os.path.join(save_dir, "boxplot_snr_vs_quality.png")
        plt.savefig(snr_quality_path, dpi=150)
        plt.close()
        print(f"✅ Boxplot gespeichert unter: {snr_quality_path}")
    else:
        print("⚠️ Keine gültigen Daten für SNR–Quality-Plot gefunden.")


    # === Scatterplot: Max Amplitude vs Quality ===
    if len(amplitudes) > 0 and len(qualities) > 0:
        plt.figure(figsize=(8, 6))
        sns.boxplot(x=qualities, y=amplitudes, palette="Set2", showfliers=False)
        sns.stripplot(x=qualities, y=amplitudes, color="k", alpha=0.5, jitter=0.2)
        plt.xlabel("Quality Class")
        plt.ylabel("Maximale Amplitude")
        plt.title("Maximale Amplitude vs. Quality Class")
        plt.grid(True, linestyle="--", alpha=0.6)
        plt.tight_layout()

        amp_quality_path = os.path.join(save_dir, "boxplot_amplitude_vs_quality.png")
        plt.savefig(amp_quality_path, dpi=150)
        plt.close()
        print(f"✅ Boxplot gespeichert unter: {amp_quality_path}")
    else:
        print("⚠️ Keine gültigen Daten für Amplitude–Quality-Plot gefunden.")

    """
    labels_internal = ["t", "m", "w"]
    # Labels for display
    display_labels = ["hmm", "moan", "wail"]

    map_classes = {"t": "hmm", "m": "moan", "w": "wail"}

    # === Scatterplot: SNR vs Max Amplitude, colored by Quality ===
    if len(snrs) > 0 and len(amplitudes) > 0 and len(qualities) > 0:
        plt.figure(figsize=(8, 6))
        sns.scatterplot(
            x=amplitudes,
            y=snrs,
            hue=qualities,          # Punkte nach Quality einfärben
            style=[map_classes[c] for c in classes],
            palette="Set3",
            markers=["o", "s", "X", "v", "D", "^", "P"], 
            alpha=0.7,
            edgecolor="k",
            s=60
        )
        # 🔴 Threshold-Linien hinzufügen
        plt.axvline(x=0.035, color='red', linestyle='--', linewidth=1.5)
        plt.axhline(y=-1,     color='red', linestyle='--', linewidth=1.5)

        #plt.xlabel("Maximale Amplitude")
        #plt.ylabel("Signal-to-Noise Ratio (dB)")
        #plt.title("SNR vs. Max Amplitude nach Quality Class")
        #plt.grid(True, linestyle="--", alpha=0.5)
        #plt.xscale("log")
        #plt.tight_layout()
        plt.xlabel("Maximal Amplitude")
        plt.ylabel("Signal-to-Noise Ratio (dB)")
        #plt.title("SNR vs. Max Amplitude by Quality and Call Class")
        plt.grid(True, linestyle="--", alpha=0.5)
        plt.xscale("log") 
        plt.xticks(rotation=45)
        #plt.yscale("log") 
        plt.tight_layout()

        snr_amp_path = os.path.join(save_dir, "scatter_snr_vs_amplitude.png")
        plt.savefig(snr_amp_path, dpi=150)
        plt.close()
        print(f"✅ Scatterplot gespeichert unter: {snr_amp_path}")
    else:
        print("⚠️ Keine gültigen Daten für SNR–Amplitude-Plot gefunden.")


    # === Scatterplot: SNR vs Max Amplitude, colored by Model Score ===
    if len(snrs) > 0 and len(amplitudes) > 0 and len(scores) > 0:
        plt.figure(figsize=(8, 6))
        scatter = plt.scatter(
            amplitudes,
            snrs,
            c=scores,             # Punkte nach Model Score einfärben
            cmap='viridis',       # Farbpalette (hell = hoch, dunkel = niedrig)
            alpha=0.8,
            edgecolor='k',
            s=60
        )
        plt.xlabel("Maximal Amplitude")
        plt.ylabel("Signal-to-Noise Ratio (dB)")
        #plt.title("SNR vs. Max Amplitude by Model Score")
        plt.grid(True, linestyle="--", alpha=0.5)
        plt.xscale("log") 
        #plt.yscale("log")
        plt.colorbar(scatter, label="Model Score")
        plt.tight_layout()

        snr_amp_score_path = os.path.join(save_dir, "scatter_snr_vs_amplitude_model_score.png")
        plt.savefig(snr_amp_score_path, dpi=150)
        plt.close()
        print(f"✅ Scatterplot gespeichert unter: {snr_amp_score_path}")
    else:
        print("⚠️ Keine gültigen Daten für SNR–Amplitude–Score-Plot gefunden.")

