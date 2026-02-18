import json
import logging
import os
import contextlib
from collections import defaultdict
from pathlib import Path

import librosa
import numpy as np
import torch
from torch.utils.data import DataLoader
from transformers import WhisperModel, WhisperFeatureExtractor, WhisperConfig

from model import WhisperFormer, infer_architecture_from_state_dict, detect_whisper_size_from_state_dict

ID_TO_CLUSTER = {
    0: "m",
    1: "h",
    2: "w",
}


def load_trained_whisperformer(checkpoint_path, device, whisper_config_path, whisper_size=None):
    """Load a trained WhisperFormer model from a .pth checkpoint.

    Architecture (num_decoder_layers, num_head_layers, num_classes) and Whisper
    size are inferred automatically from the checkpoint weights.

    Args:
        checkpoint_path: Path to the .pth checkpoint file.
        device: "cuda" or "cpu".
        whisper_config_path: Path to the whisper config directory (contains config.json).
        whisper_size: Optional explicit "base" or "large".

    Returns:
        Tuple of (model, num_classes, detected_size).
    """
    state_dict = torch.load(checkpoint_path, map_location=device)

    num_decoder_layers, num_head_layers, num_classes = infer_architecture_from_state_dict(state_dict)
    logging.info(f"Checkpoint: decoder_layers={num_decoder_layers}, head_layers={num_head_layers}, classes={num_classes}")

    if whisper_size and whisper_size.lower() in ["base", "large"]:
        detected_size = whisper_size.lower()
    else:
        detected_size = detect_whisper_size_from_state_dict(state_dict)
        if detected_size is None:
            detected_size = "large" if "large" in str(checkpoint_path).lower() else "base"

    config = WhisperConfig.from_pretrained(whisper_config_path)
    encoder = WhisperModel(config).encoder

    model = WhisperFormer(
        encoder, num_classes=num_classes,
        num_decoder_layers=num_decoder_layers,
        num_head_layers=num_head_layers,
    )

    needs_remap = any(
        k.startswith("encoder.") and not k.startswith("encoder.encoder.")
        for k in state_dict.keys()
    )
    if needs_remap:
        state_dict = {
            ("encoder." + k if k.startswith("encoder.") and not k.startswith("encoder.encoder.") else k): v
            for k, v in state_dict.items()
        }

    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()
    return model, num_classes, detected_size


def nms_1d(intervals, iou_threshold):
    """Non-maximum suppression for 1D intervals [N, 3] -> (start, end, score)."""
    if intervals.numel() == 0:
        return intervals.new_zeros((0, 3))

    starts, ends, scores = intervals[:, 0], intervals[:, 1], intervals[:, 2]
    order = scores.argsort(descending=True)
    keep = []

    while order.numel() > 0:
        i = order[0]
        keep.append(i)
        if order.numel() == 1:
            break
        ss = torch.maximum(starts[i], starts[order[1:]])
        ee = torch.minimum(ends[i], ends[order[1:]])
        inter = torch.clamp(ee - ss, min=0)
        union = (ends[i] - starts[i]) + (ends[order[1:]] - starts[order[1:]]) - inter
        iou = inter / union
        inds = (iou <= iou_threshold).nonzero(as_tuple=False).squeeze(1)
        order = order[inds + 1]

    out = intervals[torch.tensor(keep, dtype=torch.long, device=intervals.device)]
    if out.ndim == 1:
        out = out.unsqueeze(0)
    return out


def run_inference(model, audio, feature_extractor, device,
                  total_spec_columns=3000, threshold=0.35,
                  iou_threshold=0.4, batch_size=4):
    """Run WhisperFormer inference on a single audio file.

    Args:
        model: Loaded WhisperFormer model (eval mode).
        audio: 1D numpy array, 16 kHz.
        feature_extractor: WhisperFeatureExtractor instance.
        device: Torch device.
        total_spec_columns: Spectrogram columns per segment (default 3000).
        threshold: Minimum score to keep a prediction.
        iou_threshold: IoU threshold for NMS.
        batch_size: Batch size for inference.

    Returns:
        Dict with onset, offset, cluster, score lists.
    """
    sr = 16000
    spec_time_step = 0.01
    clip_duration = total_spec_columns * spec_time_step
    num_samples_in_clip = int(round(clip_duration * sr))
    sec_per_col = 0.02
    cols_per_segment = total_spec_columns // 2

    segments = []
    start = 0
    seg_idx = 0
    while start < len(audio):
        end = start + num_samples_in_clip
        clip = audio[start:end]
        if len(clip) < sr * 0.1:
            break
        clip_padded = np.concatenate([clip, np.zeros(max(0, num_samples_in_clip - len(clip)))])
        clip_padded = clip_padded.astype(np.float32)
        feats = feature_extractor(clip_padded, sampling_rate=sr, padding="do_not_pad")["input_features"][0]
        segments.append({"features": torch.tensor(feats, dtype=torch.float32), "seg_idx": seg_idx})
        start += num_samples_in_clip
        seg_idx += 1

    all_preds = {"onset": [], "offset": [], "cluster": [], "score": []}
    model.eval()

    for batch_start in range(0, len(segments), batch_size):
        batch_segs = segments[batch_start:batch_start + batch_size]
        input_features = torch.stack([s["features"] for s in batch_segs]).to(device)

        use_autocast = device != "cpu" and (
            (isinstance(device, str) and device.startswith("cuda")) or
            (hasattr(device, "type") and device.type == "cuda")
        )
        autocast_ctx = torch.amp.autocast(device_type="cuda", dtype=torch.float16) if use_autocast else contextlib.nullcontext()

        with torch.no_grad(), autocast_ctx:
            class_preds, regr_preds = model(input_features)
            class_probs = torch.sigmoid(class_preds)

        B, T, C = class_preds.shape
        for b in range(B):
            seg = batch_segs[b]
            offset_cols = seg["seg_idx"] * cols_per_segment

            for c in range(C):
                intervals = []
                for t in range(T):
                    score = class_probs[b, t, c]
                    if float(score) > threshold:
                        s = t - regr_preds[b, t, 0]
                        e = t + regr_preds[b, t, 1]
                        intervals.append(torch.stack([s, e, score]))

                if len(intervals) > 0:
                    intervals = torch.stack(intervals)
                    intervals = nms_1d(intervals, iou_threshold)
                    for start_col, end_col, sc in intervals.cpu().tolist():
                        all_preds["onset"].append(float((offset_cols + start_col) * sec_per_col))
                        all_preds["offset"].append(float((offset_cols + end_col) * sec_per_col))
                        all_preds["cluster"].append(ID_TO_CLUSTER.get(c, "unknown"))
                        all_preds["score"].append(float(sc))

    return all_preds


def infer(data_dir, checkpoint_path, whisper_config_path, output_dir,
          threshold=0.35, iou_threshold=0.4, total_spec_columns=3000,
          batch_size=4, device=None):
    """Run WhisperFormer inference on all WAV files in a directory.

    Args:
        data_dir: Folder containing .wav files.
        checkpoint_path: Path to the .pth checkpoint.
        whisper_config_path: Path to the whisper config directory.
        output_dir: Folder to write prediction .json files.
        threshold: Score threshold.
        iou_threshold: NMS IoU threshold.
        total_spec_columns: Spectrogram columns per segment.
        batch_size: Inference batch size.
        device: "cuda" or "cpu" (auto-detected if None).
    """
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    model, num_classes, detected_size = load_trained_whisperformer(
        checkpoint_path, device, whisper_config_path
    )
    logging.info(f"Model loaded ({num_classes} classes, Whisper {detected_size}) on {device}")

    feature_extractor = WhisperFeatureExtractor.from_pretrained(
        whisper_config_path, local_files_only=True
    )

    files = sorted(Path(data_dir).rglob("*.[Ww][Aa][Vv]"))
    logging.info(f"Found {len(files)} WAV files in: {data_dir}")

    os.makedirs(output_dir, exist_ok=True)

    for i, wav_path in enumerate(files, start=1):
        logging.info(f"[{i}/{len(files)}] Processing: {wav_path.name}")

        audio, _ = librosa.load(wav_path, sr=16000)

        preds = run_inference(
            model, audio, feature_extractor, device,
            total_spec_columns=total_spec_columns,
            threshold=threshold,
            iou_threshold=iou_threshold,
            batch_size=batch_size,
        )

        out_path = os.path.join(output_dir, wav_path.stem + ".json")
        with open(out_path, "w") as f:
            json.dump(preds, f, indent=2)

        logging.info(f"Saved {len(preds['onset'])} predictions to {out_path}")

    logging.info("All files processed.")
