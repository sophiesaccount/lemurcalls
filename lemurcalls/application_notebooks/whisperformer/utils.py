import json
import logging
import os
import contextlib
from collections import defaultdict
from pathlib import Path

import librosa
import numpy as np
import torch
from transformers import WhisperModel, WhisperFeatureExtractor, WhisperConfig

from model import (
    WhisperFormer,
    infer_architecture_from_state_dict,
    detect_whisper_size_from_state_dict,
)

ID_TO_CLUSTER_3 = {0: "m", 1: "h", 2: "w"}
ID_TO_CLUSTER_1 = {0: "m"}


def get_id_to_cluster(num_classes):
    if num_classes == 3:
        return ID_TO_CLUSTER_3
    elif num_classes == 1:
        return ID_TO_CLUSTER_1
    else:
        return {i: str(i) for i in range(num_classes)}


def load_trained_whisperformer(
    checkpoint_path, device, whisper_config_path, whisper_size=None
):
    """Load a trained WhisperFormer model from a .pth checkpoint.

    Architecture (num_decoder_layers, num_head_layers, num_classes) and Whisper
    size are inferred automatically from the checkpoint weights.

    Args:
        checkpoint_path: Path to the .pth checkpoint file.
        device: "cuda" or "cpu".
        whisper_config_path: Path to the whisper config directory (must contain config.json).
        whisper_size: Optional explicit "base" or "large".

    Returns:
        Tuple of (model, num_classes, detected_size).
    """
    state_dict = torch.load(checkpoint_path, map_location=device, weights_only=True)

    num_decoder_layers, num_head_layers, num_classes = (
        infer_architecture_from_state_dict(state_dict)
    )
    logging.info(
        f"Checkpoint: decoder_layers={num_decoder_layers}, "
        f"head_layers={num_head_layers}, classes={num_classes}"
    )

    if whisper_size and whisper_size.lower() in ("base", "large"):
        detected_size = whisper_size.lower()
    else:
        detected_size = detect_whisper_size_from_state_dict(state_dict)
        if detected_size is None:
            detected_size = (
                "large" if "large" in str(checkpoint_path).lower() else "base"
            )

    config = WhisperConfig.from_pretrained(whisper_config_path)
    encoder = WhisperModel(config).encoder

    model = WhisperFormer(
        encoder,
        num_classes=num_classes,
        num_decoder_layers=num_decoder_layers,
        num_head_layers=num_head_layers,
    )

    needs_remap = any(
        k.startswith("encoder.") and not k.startswith("encoder.encoder.")
        for k in state_dict
    )
    if needs_remap:
        state_dict = {
            (
                "encoder." + k
                if k.startswith("encoder.") and not k.startswith("encoder.encoder.")
                else k
            ): v
            for k, v in state_dict.items()
        }

    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()
    return model, num_classes, detected_size


# ---------------------------------------------------------------------------
# NMS
# ---------------------------------------------------------------------------


def nms_1d(intervals, iou_threshold):
    """Non-maximum suppression for 1-D intervals.

    Args:
        intervals: Tensor of shape [N, 3] with columns (start, end, score).
        iou_threshold: Suppress intervals with IoU above this value.

    Returns:
        Filtered intervals tensor [K, 3].
    """
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
        iou = inter / (union + 1e-8)
        mask = (iou <= iou_threshold).nonzero(as_tuple=False).squeeze(1)
        order = order[mask + 1]

    out = intervals[torch.tensor(keep, dtype=torch.long, device=intervals.device)]
    return out.unsqueeze(0) if out.ndim == 1 else out


# ---------------------------------------------------------------------------
# Single-pass inference on one audio array
# ---------------------------------------------------------------------------


def _run_inference_single_pass(
    model,
    audio,
    feature_extractor,
    device,
    total_spec_columns,
    threshold,
    iou_threshold,
    batch_size,
    id_to_cluster,
    sample_offset=0,
):
    """Run inference on one audio array with a given sample offset.

    Returns list of dicts with onset/offset/cluster/score per prediction.
    """
    sr = 16000
    spec_time_step = 0.01
    clip_duration = total_spec_columns * spec_time_step
    num_samples_in_clip = int(round(clip_duration * sr))
    sec_per_col = 0.02
    cols_per_segment = total_spec_columns // 2

    segments = []
    start = sample_offset
    seg_idx = 0
    while start < len(audio):
        end = start + num_samples_in_clip
        clip = audio[start:end]
        if len(clip) < sr * 0.1:
            break
        clip_padded = np.concatenate(
            [clip, np.zeros(max(0, num_samples_in_clip - len(clip)))]
        )
        clip_padded = clip_padded.astype(np.float32)
        feats = feature_extractor(clip_padded, sampling_rate=sr, padding="do_not_pad")[
            "input_features"
        ][0]
        segments.append(
            {
                "features": torch.tensor(feats, dtype=torch.float32),
                "seg_idx": seg_idx,
                "start_sec": start / sr,
            }
        )
        start += num_samples_in_clip
        seg_idx += 1

    if not segments:
        return []

    use_autocast = device != "cpu" and (
        (isinstance(device, str) and device.startswith("cuda"))
        or (hasattr(device, "type") and device.type == "cuda")
    )
    autocast_ctx = (
        torch.amp.autocast(device_type="cuda", dtype=torch.float16)
        if use_autocast
        else contextlib.nullcontext()
    )

    predictions = []

    for batch_start in range(0, len(segments), batch_size):
        batch_segs = segments[batch_start : batch_start + batch_size]
        input_features = torch.stack([s["features"] for s in batch_segs]).to(device)

        with torch.no_grad(), autocast_ctx:
            class_preds, regr_preds = model(input_features)
            class_probs = torch.sigmoid(class_preds)

        B, T, C = class_preds.shape
        for b in range(B):
            seg = batch_segs[b]
            offset_sec = seg["start_sec"]

            for c in range(C):
                intervals = []
                for t in range(T):
                    score = class_probs[b, t, c]
                    if float(score) > threshold:
                        s = t - regr_preds[b, t, 0]
                        e = t + regr_preds[b, t, 1]
                        intervals.append(torch.stack([s, e, score]))

                if intervals:
                    intervals = torch.stack(intervals)
                    intervals = nms_1d(intervals, iou_threshold)
                    for start_col, end_col, sc in intervals.cpu().tolist():
                        predictions.append(
                            {
                                "onset": offset_sec + start_col * sec_per_col,
                                "offset": offset_sec + end_col * sec_per_col,
                                "cluster": id_to_cluster.get(c, "unknown"),
                                "score": float(sc),
                            }
                        )

    return predictions


# ---------------------------------------------------------------------------
# Consolidation across multiple offset runs
# ---------------------------------------------------------------------------


def _consolidate(all_runs_preds, overlap_tolerance=0.1):
    """Merge overlapping predictions from multiple runs; keep highest-score per group.

    Only predictions confirmed by at least 2 runs are kept.
    """
    if not all_runs_preds:
        return {"onset": [], "offset": [], "cluster": [], "score": []}

    tagged = []
    for run_idx, preds in enumerate(all_runs_preds):
        for p in preds:
            tagged.append({**p, "run": run_idx})

    clusters = set(p["cluster"] for p in tagged)
    result = {"onset": [], "offset": [], "cluster": [], "score": []}

    for cluster in clusters:
        cpreds = [p for p in tagged if p["cluster"] == cluster]
        used = set()
        for i, p1 in enumerate(cpreds):
            if i in used:
                continue
            group = [p1]
            runs_seen = {p1["run"]}

            for j, p2 in enumerate(cpreds):
                if i == j or j in used:
                    continue
                inter = max(
                    0, min(p1["offset"], p2["offset"]) - max(p1["onset"], p2["onset"])
                )
                union = max(p1["offset"], p2["offset"]) - min(p1["onset"], p2["onset"])
                iou = inter / union if union > 0 else 0
                if iou > overlap_tolerance:
                    group.append(p2)
                    runs_seen.add(p2["run"])
                    used.add(j)

            if len(runs_seen) >= 2:
                best = max(group, key=lambda x: x["score"])
                result["onset"].append(best["onset"])
                result["offset"].append(best["offset"])
                result["cluster"].append(best["cluster"])
                result["score"].append(best["score"])

    return result


# ---------------------------------------------------------------------------
# High-level inference on a single audio
# ---------------------------------------------------------------------------


def run_inference(
    model,
    audio,
    feature_extractor,
    device,
    id_to_cluster,
    total_spec_columns=3000,
    threshold=0.35,
    iou_threshold=0.4,
    batch_size=4,
    num_runs=3,
    overlap_tolerance=0.1,
):
    """Run WhisperFormer inference on a single audio (multi-offset + consolidation).

    The audio is processed multiple times with different starting offsets.
    Predictions confirmed by at least 2 runs are kept (consolidated).

    Args:
        model: Loaded WhisperFormer (eval mode).
        audio: 1-D numpy array, 16 kHz.
        feature_extractor: WhisperFeatureExtractor instance.
        device: Torch device.
        id_to_cluster: Dict mapping class id -> cluster name.
        total_spec_columns: Input spectrogram columns per segment (default 3000 = 30 s).
        threshold: Minimum score to keep a prediction.
        iou_threshold: IoU threshold for NMS within each run.
        batch_size: Batch size for inference.
        num_runs: Number of offset runs (1 = single pass, 3 = default with offsets 0/1000/2000).
        overlap_tolerance: IoU threshold for consolidation across runs.

    Returns:
        Dict with onset, offset, cluster, score lists.
    """
    shift_offsets = [i * 1000 for i in range(num_runs)]

    if num_runs == 1:
        preds = _run_inference_single_pass(
            model,
            audio,
            feature_extractor,
            device,
            total_spec_columns,
            threshold,
            iou_threshold,
            batch_size,
            id_to_cluster,
            sample_offset=0,
        )
        return {
            "onset": [p["onset"] for p in preds],
            "offset": [p["offset"] for p in preds],
            "cluster": [p["cluster"] for p in preds],
            "score": [p["score"] for p in preds],
        }

    all_runs = []
    for offset in shift_offsets:
        preds = _run_inference_single_pass(
            model,
            audio,
            feature_extractor,
            device,
            total_spec_columns,
            threshold,
            iou_threshold,
            batch_size,
            id_to_cluster,
            sample_offset=offset,
        )
        all_runs.append(preds)

    return _consolidate(all_runs, overlap_tolerance=overlap_tolerance)


# ---------------------------------------------------------------------------
# Top-level: process all WAV files in a directory
# ---------------------------------------------------------------------------


def infer(
    data_dir,
    checkpoint_path,
    whisper_config_path,
    output_dir,
    threshold=0.35,
    iou_threshold=0.4,
    total_spec_columns=3000,
    batch_size=4,
    num_runs=3,
    overlap_tolerance=0.1,
    device=None,
):
    """Run WhisperFormer inference on all WAV files in a directory.

    Args:
        data_dir: Folder containing .wav files.
        checkpoint_path: Path to the .pth checkpoint.
        whisper_config_path: Path to the whisper config directory.
        output_dir: Folder to write prediction .json files.
        threshold: Score threshold (default 0.35).
        iou_threshold: NMS IoU threshold (default 0.4).
        total_spec_columns: Spectrogram columns per segment (default 3000).
        batch_size: Inference batch size (default 4).
        num_runs: Number of offset runs for consolidation (default 3).
        overlap_tolerance: IoU threshold for cross-run consolidation (default 0.1).
        device: "cuda" or "cpu" (auto-detected if None).

    Returns:
        Dict mapping filename -> prediction dict.
    """
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    model, num_classes, detected_size = load_trained_whisperformer(
        checkpoint_path, device, whisper_config_path
    )
    id_to_cluster = get_id_to_cluster(num_classes)
    logging.info(
        f"Model loaded: {num_classes} class(es), Whisper {detected_size}, "
        f"device={device}, classes={list(id_to_cluster.values())}"
    )

    feature_extractor = WhisperFeatureExtractor.from_pretrained(
        whisper_config_path, local_files_only=True
    )

    files = sorted(Path(data_dir).rglob("*.[Ww][Aa][Vv]"))
    logging.info(f"Found {len(files)} WAV file(s) in: {data_dir}")
    if not files:
        logging.warning("No WAV files found. Nothing to process.")
        return {}

    os.makedirs(output_dir, exist_ok=True)
    results = {}

    for i, wav_path in enumerate(files, start=1):
        logging.info(f"[{i}/{len(files)}] Processing: {wav_path.name}")

        audio, _ = librosa.load(wav_path, sr=16000)

        preds = run_inference(
            model,
            audio,
            feature_extractor,
            device,
            id_to_cluster,
            total_spec_columns=total_spec_columns,
            threshold=threshold,
            iou_threshold=iou_threshold,
            batch_size=batch_size,
            num_runs=num_runs,
            overlap_tolerance=overlap_tolerance,
        )

        out_path = os.path.join(output_dir, wav_path.stem + ".json")
        with open(out_path, "w") as f:
            json.dump(preds, f, indent=2)

        n = len(preds["onset"])
        logging.info(f"  -> {n} prediction(s) saved to {out_path}")
        results[wav_path.name] = preds

    logging.info("All files processed.")
    return results
