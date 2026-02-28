import os
import json
import argparse
import librosa
import numpy as np
import soundfile as sf
import matplotlib.pyplot as plt
from tqdm import tqdm
from datetime import datetime
from scipy.signal import butter, filtfilt
from ..visualization.scatterplot_ampl_snr_score import compute_snr_new
from ...datautils import get_audio_and_label_paths_from_folders


def highpass_filter(y, sr, cutoff=200, order=5):
    """Apply Butterworth highpass filter."""
    nyquist = 0.5 * sr
    normal_cutoff = cutoff / nyquist
    b, a = butter(order, normal_cutoff, btype="high", analog=False)
    y_filtered = filtfilt(b, a, y)
    return y_filtered


def compute_snr(y, sr, cutoff=200, order=5):
    """Compute simple SNR (signal = >cutoff, noise = <cutoff)."""
    y_signal = highpass_filter(y, sr, cutoff=cutoff, order=order)
    nyquist = 0.5 * sr
    normal_cutoff = cutoff / nyquist
    b, a = butter(order, normal_cutoff, btype="low", analog=False)
    y_noise = filtfilt(b, a, y)

    signal_power = np.mean(y_signal**2) + 1e-10
    noise_power = np.mean(y_noise**2) + 1e-10
    snr_db = 10 * np.log10(signal_power / noise_power)
    return snr_db


def main():
    parser = argparse.ArgumentParser(
        description="Extract and save labeled audio segments above an SNR threshold."
    )
    parser.add_argument(
        "--audio_folder", required=True, help="Path to folder containing .wav files."
    )
    parser.add_argument(
        "--label_folder",
        required=True,
        help="Path to folder containing .json label files.",
    )
    parser.add_argument(
        "--output_dir",
        default="high_snr_segments",
        help="Directory to save high-SNR snippets.",
    )
    parser.add_argument(
        "--snr_threshold", type=float, default=-1, help="SNR threshold in dB."
    )
    parser.add_argument(
        "--sample_rate",
        type=int,
        default=16000,
        help="Target sample rate for loading audio.",
    )
    parser.add_argument(
        "--cutoff",
        type=float,
        default=200,
        help="Frequency cutoff for SNR computation (Hz).",
    )
    parser.add_argument(
        "--no_plot", action="store_true", help="Disable histogram plotting."
    )
    parser.add_argument(
        "--amplitude_threshold",
        type=float,
        default=0.035,
        help="Minimum peak amplitude for keeping a segment (0–1 range).",
    )

    args = parser.parse_args()

    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    save_dir = os.path.join(
        args.output_dir, f"snr_above_{args.snr_threshold}dB_{timestamp}"
    )
    os.makedirs(save_dir, exist_ok=True)

    audio_files = sorted(
        [f for f in os.listdir(args.audio_folder) if f.endswith(".WAV")]
    )
    label_files = sorted(
        [f for f in os.listdir(args.label_folder) if f.endswith((".jsonr", ".json"))]
    )

    if len(audio_files) == 0:
        print("No .wav files found in", args.audio_folder)
        return
    if len(label_files) == 0:
        print("No .json label files found in", args.label_folder)
        return

    print(f"Found {len(audio_files)} audio files.")
    print(f"Found {len(label_files)} label files.")

    count_saved = 0
    all_snrs = []

    audio_paths, label_paths = get_audio_and_label_paths_from_folders(
        args.audio_folder, args.label_folder
    )
    print(label_paths)
    for audio_path, label_path in tqdm(
        zip(audio_paths, label_paths), total=len(label_paths)
    ):
        print(audio_path)
        print(label_path)

        # Load audio
        y, sr = librosa.load(audio_path, sr=args.sample_rate)

        # Load labels
        with open(label_path, "r") as f:
            labels = json.load(f)

        onsets = labels.get("onset", [])
        offsets = labels.get("offset", [])
        clusters = labels.get("cluster", [])
        qualities = labels.get("quality", ["unknown"] * len(onsets))
        scores = labels.get("score", ["unknown"] * len(onsets))

        # Prepare filtered lists
        (
            kept_onsets,
            kept_offsets,
            kept_clusters,
            kept_qualities,
            kept_scores,
            kept_snrs,
        ) = [], [], [], [], [], []

        for onset, offset, cluster_label, quality_label, score_label in zip(
            onsets, offsets, clusters, qualities, scores
        ):
            start_sample = int(onset * sr)
            end_sample = int(offset * sr)
            segment_audio = y[start_sample:end_sample]

            if len(segment_audio) < 40:
                continue

            snr_db = compute_snr_new(
                segment_audio, sr, cutoff=args.cutoff, signal_high=1000
            )
            segment_audio_filtered = highpass_filter(
                segment_audio, sr, cutoff=args.cutoff
            )
            max_amplitude = float(np.max(np.abs(segment_audio_filtered)))
            all_snrs.append(snr_db)
            print(args.amplitude_threshold)
            if snr_db > args.snr_threshold and max_amplitude > args.amplitude_threshold:
                # Keep this label
                kept_onsets.append(onset)
                kept_offsets.append(offset)
                kept_clusters.append(cluster_label)
                kept_qualities.append(quality_label)
                kept_scores.append(score_label)
                kept_snrs.append(float(snr_db))

                # Optionally save the audio snippet
                base_name = os.path.splitext(os.path.basename(audio_path))[0]
                snippet_name = f"{base_name}_{cluster_label}_{onset:.2f}-{offset:.2f}s_SNR{snr_db:.1f}.wav"
                snippet_path = os.path.join(save_dir, snippet_name)
                sf.write(snippet_path, segment_audio, sr)
                count_saved += 1

        # === Save filtered JSON ===
        if len(kept_onsets) > 0:
            filtered_labels = {
                "onset": kept_onsets,
                "offset": kept_offsets,
                "cluster": kept_clusters,
                "quality": kept_qualities,
                "score": kept_scores,
                "snr_db": kept_snrs,
            }
            filtered_json_name = (
                os.path.splitext(os.path.basename(audio_path))[0] + ".json"
            )
            filtered_json_path = os.path.join(save_dir, filtered_json_name)
            with open(filtered_json_path, "w") as jf:
                json.dump(filtered_labels, jf, indent=2)

                count_saved += 1

    print(
        f"\nDone. Saved {count_saved} high-SNR segments (> {args.snr_threshold} dB) in: {save_dir}"
    )

    # Optional histogram plot
    if not args.no_plot and len(all_snrs) > 0:
        plt.figure(figsize=(8, 5))
        plt.hist(all_snrs, bins=40, color="skyblue", edgecolor="black", alpha=0.8)
        plt.axvline(
            args.snr_threshold,
            color="r",
            linestyle="--",
            label=f"Threshold = {args.snr_threshold} dB",
        )
        plt.title("Distribution of Segment SNRs")
        plt.xlabel("SNR (dB)")
        plt.ylabel("Count")
        plt.legend()
        plt.grid(True, linestyle="--", alpha=0.5)
        plt.tight_layout()

        hist_path = os.path.join(save_dir, "snr_histogram.png")
        plt.savefig(hist_path, dpi=150)
        plt.close()

        print(f"Histogram saved to: {hist_path}")
    elif len(all_snrs) == 0:
        print("No valid SNR values found; no histogram created.")


if __name__ == "__main__":
    main()
