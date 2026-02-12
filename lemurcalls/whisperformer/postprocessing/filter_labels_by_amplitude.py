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
parser.add_argument("--amplitude_threshold", type=float, required=True, help="Amplitude-Threshold (z.B. 0.05)")
args = parser.parse_args()

wav_path = args.wav_path
labels_path = args.labels_path
output_path = args.output_path
amplitude_threshold = args.amplitude_threshold

# === Audio und Labels laden ===
audio, sr = librosa.load(wav_path, sr=None)
with open(labels_path, "r") as f:
    labels = json.load(f)

onsets = np.array(labels["onset"])
offsets = np.array(labels["offset"])
clusters = labels["cluster"]

# === Filtere Labels nach Amplitude ===
filtered_onsets = []
filtered_offsets = []
filtered_clusters = []

for onset, offset, cluster in zip(onsets, offsets, clusters):
    audio_segment = audio[int(onset * sr):int(offset * sr)]
    max_amp = np.max(np.abs(audio_segment))
    if max_amp > amplitude_threshold:
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
print(f"Anzahl der übrig gebliebenen Labels: {len(filtered_onsets)}") 