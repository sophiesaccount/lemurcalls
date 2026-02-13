import json
import librosa
import numpy as np
import os
import argparse

parser = argparse.ArgumentParser(description="Filter labels by amplitude threshold for a whole directory.")
parser.add_argument("--wav_dir", required=True, help="Path to WAV directory")
parser.add_argument("--labels_dir", required=True, help="Path to labels directory")
parser.add_argument("--output_dir", required=True, help="Path to output directory")
parser.add_argument("--amplitude_threshold", type=float, required=True, help="Amplitude threshold (e.g. 0.05)")
args = parser.parse_args()

wav_dir = args.wav_dir
labels_dir = args.labels_dir
output_dir = args.output_dir
amplitude_threshold = args.amplitude_threshold

for root, _, files in os.walk(labels_dir):
    for file in files:
        if not file.endswith('.json'):
            continue
        label_path = os.path.join(root, file)
        rel_path = os.path.relpath(label_path, labels_dir)
        wav_path = os.path.join(wav_dir, os.path.splitext(rel_path)[0] + ".wav")
        output_path = os.path.join(output_dir, rel_path)

        if not os.path.exists(wav_path):
            print(f"WAV file not found for {label_path}, skipping.")
            continue

        # Load audio and labels
        audio, sr = librosa.load(wav_path, sr=None)
        with open(label_path, "r") as f:
            labels = json.load(f)

        onsets = np.array(labels["onset"])
        offsets = np.array(labels["offset"])
        clusters = labels["cluster"]

        # Filter labels by amplitude
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

        # Ensure output directory exists
        os.makedirs(os.path.dirname(output_path), exist_ok=True)

        # Save filtered labels
        with open(output_path, "w") as f:
            json.dump(filtered_labels, f, indent=2)

        print(f"Filtered labels saved to: {output_path}")
        print(f"Remaining labels: {len(filtered_onsets)}")