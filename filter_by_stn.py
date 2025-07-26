import json
import librosa
import numpy as np
import os
import argparse

# === Argumente parsen ===
parser = argparse.ArgumentParser(description="Filtere Labels nach Amplituden-Threshold.")
parser.add_argument("--wav_path", required=True, help="Pfad zur WAV-Datei")
parser.add_argument("--labels_path", required=True, help="Pfad zur Labels-JSON")
parser.add_argument("--output_path", required=True, help="Pfad zur Ausgabe-JSON")
parser.add_argument("--stn_threshold", type=float, required=True, help="Signal-to-Noise-Threshold (z.B. 5)")
args = parser.parse_args()

wav_path = args.wav_path
labels_path = args.labels_path
output_path = args.output_path
stn_threshold = args.stn_threshold

# === Audio und Labels laden ===
audio, sr = librosa.load(wav_path, sr=None)
with open(labels_path, "r") as f:
    labels = json.load(f)

onsets = np.array(labels["onset"])
offsets = np.array(labels["offset"])
clusters = labels["cluster"]



def calculate_snr(audio, sr, onset, offset, noise_window=0.1, noise_gap=0.01):
    """
    audio: full audio array
    sr: sample rate
    onset, offset: segment start/end in seconds
    noise_window: duration (s) before onset to use as noise
    """
    # Signal segment
    start_sample = int(onset * sr)
    end_sample = int(offset * sr)
    signal = audio[start_sample:end_sample]

    # Noise segment (before onset, mit Abstand noise_gap)
    noise_end = max(0, start_sample - int(noise_gap * sr))
    noise_start = max(0, noise_end - int(noise_window * sr))
    noise = audio[noise_start:noise_end]
    
    """
    # Noise segment (before onset)
    noise_end = max(0, start_sample)
    noise_start = max(0, noise_end - int(noise_window * sr))
    noise = audio[noise_start:noise_end]
    """

    # RMS calculation
    rms_signal = np.sqrt(np.mean(signal**2)) if len(signal) > 0 else 1e-12
    rms_noise = np.sqrt(np.mean(noise**2)) if len(noise) > 0 else 1e-12
    
    # SNR in dB
    snr_db = 20 * np.log10(rms_signal / rms_noise)
    print(f"SNR: {snr_db}")
    return snr_db


# === Filtere Labels nach Amplitude ===
filtered_onsets = []
filtered_offsets = []
filtered_clusters = []

for onset, offset, cluster in zip(onsets, offsets, clusters):
    #audio_segment = audio[int(onset * sr):int(offset * sr)]
    stn_ratio = calculate_snr(audio, sr, onset, offset, noise_window=0.1, noise_gap=0.1)
    if stn_ratio > stn_threshold:
        filtered_onsets.append(onset)
        filtered_offsets.append(offset)
        filtered_clusters.append(cluster)

filtered_labels = {
    "onset": filtered_onsets,
    "offset": filtered_offsets,
    "cluster": filtered_clusters
}

# === Stelle sicher, dass der Zielordner existiert ===
os.makedirs(os.path.dirname(output_path), exist_ok=True)

# === Speichere die gefilterten Labels ===
with open(output_path, "w") as f:
    json.dump(filtered_labels, f, indent=2)

print(f"Gefilterte Labels gespeichert in: {output_path}")
print(f"Anzahl der ursprünglichen Labels: {len(onsets)}")
print(f"Anzahl der übrig gebliebenen Labels: {len(filtered_onsets)}") 