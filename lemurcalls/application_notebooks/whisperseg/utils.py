import os
import logging
import sys
from os.path import join
from pathlib import Path
import json
import argparse

from scipy.io import wavfile
import librosa  # FEHLT
import numpy as np  # optional, falls intern benötigt
import torch  # optional, falls WhisperSegmenterFast Torch nutzt

from model import WhisperSegmenterFast



def create_if_not_exists(folder):
    if not os.path.exists(folder):
        os.makedirs( folder )
    return folder

def get_lr(optimizer):
    return [param_group['lr'] for param_group in optimizer.param_groups ]


def fetch_sr_rounded(wav_path: str) -> int:
    """Fetches the sampling rate from a wav file and rounds it to the nearest 10s value (22050 -> 22000).

    Args:
        wav_path (str): Path to the wav file

    Returns:
        int: The rounded sampling rate
    """
    return round(
        wavfile.read(wav_path)[0], # [sampling_rate, data]
        -2)



def infer(data_dir: str, model_path: str, output_dir: str, min_frequency: int, spec_time_step: float,
          min_segment_length: float, eps: float, num_trials: int):

    # Logging konfigurieren
    logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s')

    segmenter = WhisperSegmenterFast(model_path, device="cpu")
    logging.info("Model loaded successfully.")

    # Alle WAV-Dateien finden (case-insensitive)
    files = list(Path(data_dir).rglob("*.[Ww][Aa][Vv]"))
    num_files = len(files)
    logging.info(f"Found {num_files} wav files in: {data_dir}")

    for i, p in enumerate(files, start=1):
        logging.info(f"[{i}/{num_files}] Processing file: {p.name}")

        sampling_rate = fetch_sr_rounded(p.absolute())
        audio, _ = librosa.load(p, sr=sampling_rate)

        prediction = segmenter.segment(
            audio,
            sr=sampling_rate,
            min_frequency=min_frequency,
            spec_time_step=spec_time_step,
            min_segment_length=min_segment_length,
            eps=eps,
            num_trials=num_trials
        )

        # Sicherstellen, dass Output-Ordner existiert
        if output_dir is None:
            new_path = p.parent.absolute()
        else:
            new_path = output_dir
        os.makedirs(new_path, exist_ok=True)

        out_path = join(new_path, p.stem + '.json')  # besser .json statt .jsonr
        with open(out_path, "w") as fp:
            json.dump(prediction, fp=fp, indent=2)

        logging.info(f"✅ Finished file {i}/{num_files}, saved to {out_path}")

    logging.info("All files processed.")
