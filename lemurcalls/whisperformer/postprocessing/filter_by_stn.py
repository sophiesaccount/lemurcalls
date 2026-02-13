import json
import librosa
import numpy as np
import os
import argparse

parser = argparse.ArgumentParser(description="Filter labels by signal-to-noise threshold.")
parser.add_argument("--wav_path", required=True, help="Path to WAV file")
parser.add_argument("--labels_path", required=True, help="Path to labels JSON")
parser.add_argument("--output_path", required=True, help="Path to output JSON")
parser.add_argument("--stn_threshold", type=float, required=True, help="Signal-to-noise threshold (e.g. 5)")
args = parser.parse_args()

wav_path = args.wav_path
labels_path = args.labels_path
output_path = args.output_path
stn_threshold = args.stn_threshold

audio, sr = librosa.load(wav_path, sr=None)
with open(labels_path, "r") as f:
    labels = json.load(f)

onsets = np.array(labels["onset"])
offsets = np.array(labels["offset"])
clusters = labels["cluster"]



def calculate_snr(audio, sr, onset, offset, noise_window=0.1, noise_gap=0.01):
    """Compute SNR in dB for a segment; noise is taken from before onset (with gap).

    Args:
        audio: Full audio array.
        sr: Sample rate.
        onset, offset: Segment start/end in seconds.
        noise_window: Duration (s) of noise window before onset.
        noise_gap: Gap (s) between segment start and noise window end.

    Returns:
        SNR in dB.
    """
    start_sample = int(onset * sr)
    end_sample = int(offset * sr)
    signal = audio[start_sample:end_sample]
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
    snr_db = 20 * np.log10(rms_signal / rms_noise)
    return snr_db


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

if os.path.dirname(output_path):
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
with open(output_path, "w") as f:
    json.dump(filtered_labels, f, indent=2)

print(f"Filtered labels saved to: {output_path}")
print(f"Original number of labels: {len(onsets)}")
print(f"Remaining labels: {len(filtered_onsets)}") 