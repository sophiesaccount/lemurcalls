import json
import logging
import os
from os.path import join
from pathlib import Path

import librosa
from scipy.io import wavfile

from model import WhisperSegmenterFast


def fetch_sr_rounded(wav_path: str) -> int:
    """Fetch the sampling rate from a wav file, rounded to the nearest 100.

    E.g. 22050 -> 22000, 44100 -> 44100.

    Args:
        wav_path: Path to the wav file.

    Returns:
        Rounded sampling rate (int).
    """
    return round(wavfile.read(wav_path)[0], -2)


def infer(
    data_dir: str,
    model_path: str,
    output_dir: str,
    min_frequency: int,
    spec_time_step: float,
    min_segment_length: float,
    eps: float,
    num_trials: int,
):
    """Run WhisperSeg inference on all WAV files in a directory.

    Args:
        data_dir: Folder containing .wav files.
        model_path: Path to the CT2 model directory.
        output_dir: Folder to write prediction .json files.
        min_frequency: Minimum frequency for spectrogram (Hz).
        spec_time_step: Time step for spectrogram (seconds).
        min_segment_length: Minimum segment length (seconds).
        eps: DBSCAN epsilon for clustering.
        num_trials: Number of segmentation trials.

    Returns:
        Dict mapping filename -> prediction dict.
    """
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    segmenter = WhisperSegmenterFast(model_path, device="cpu")
    logging.info("Model loaded successfully.")

    files = sorted(Path(data_dir).rglob("*.[Ww][Aa][Vv]"))
    num_files = len(files)
    logging.info(f"Found {num_files} WAV file(s) in: {data_dir}")

    if not files:
        logging.warning("No WAV files found. Nothing to process.")
        return {}

    os.makedirs(output_dir, exist_ok=True)
    results = {}

    for i, p in enumerate(files, start=1):
        logging.info(f"[{i}/{num_files}] Processing: {p.name}")

        sampling_rate = fetch_sr_rounded(str(p.absolute()))
        audio, _ = librosa.load(p, sr=sampling_rate)

        prediction = segmenter.segment(
            audio,
            sr=sampling_rate,
            min_frequency=min_frequency,
            spec_time_step=spec_time_step,
            min_segment_length=min_segment_length,
            eps=eps,
            num_trials=num_trials,
        )

        out_path = join(output_dir, p.stem + ".json")
        with open(out_path, "w") as fp:
            json.dump(prediction, fp=fp, indent=2)

        n_preds = len(prediction.get("onset", []))
        logging.info(f"  -> {n_preds} prediction(s) saved to {out_path}")
        results[p.name] = prediction

    logging.info("All files processed.")
    return results
