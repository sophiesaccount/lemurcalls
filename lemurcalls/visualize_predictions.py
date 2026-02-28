import argparse
import json
import os
import contextlib
import torch
from torch.utils.data import DataLoader
from datetime import datetime
from collections import defaultdict
import numpy as np

from .whisperformer.dataset import WhisperFormerDatasetQuality
from .whisperformer.model import WhisperFormer
from transformers import WhisperModel, WhisperFeatureExtractor, WhisperConfig
from .datautils import (
    get_audio_and_label_paths_from_folders,
    load_data,
    slice_audios_and_labels,
    FIXED_CLUSTER_CODEBOOK,
    ID_TO_CLUSTER,
)
from .whisperformer.train import collate_fn, nms_1d_torch
from .whisperformer.visualization.scatterplot_ampl_snr_score import (
    compute_snr_new,
    compute_snr_timebased,
)

import librosa.display

import librosa.display
import numpy as np
import os
import matplotlib.patches as mpatches
import glob


def plot_spectrogram_and_scores(
    mel_spec,
    class_scores,
    gt_onsets,
    gt_offsets,
    gt_classes,
    gt_qualities,
    pred_onsets,
    pred_offsets,
    pred_classes,
    segment_idx,
    save_dir,
    base_name,
    y,
    threshold=0.35,
    ID_TO_CLUSTER=None,
    extra_onsets=None,
    extra_offsets=None,
    extra_labels=None,
    extra_title="WhisperSeg Predictions",
    gt_title="GT Labels with Quality Classes",
):
    """Plot mel spectrogram with model scores, ground truth, and optional extra labels.

    Generates a multi-row figure containing the mel spectrogram, per-class
    score bars with predicted segments, ground truth label spans (with quality
    annotations), and optionally a fourth row for additional label spans
    (e.g. WhisperSeg predictions for comparison).

    Args:
        mel_spec: Mel spectrogram array of shape ``(n_mels, T)``.
        class_scores: Per-frame class scores of shape ``(T, num_classes)``.
        gt_onsets: Ground truth onset times in seconds.
        gt_offsets: Ground truth offset times in seconds.
        gt_classes: Ground truth class labels (cluster strings).
        gt_qualities: Quality ratings for each ground truth label (1, 2, 3, ...).
        pred_onsets: Predicted onset times in seconds.
        pred_offsets: Predicted offset times in seconds.
        pred_classes: Predicted class labels (cluster strings).
        segment_idx: Index of the current segment (used in the filename).
        save_dir: Directory to save the output figure.
        base_name: Base filename for the saved plot.
        y: Raw audio waveform for the segment (used for SNR computation).
        threshold: Score threshold for the horizontal reference line.
        ID_TO_CLUSTER: Mapping from class ID to cluster label string.
        extra_onsets: Optional extra onset times (e.g. from another model).
        extra_offsets: Optional extra offset times.
        extra_labels: Optional extra class labels.
        extra_title: Title for the extra labels row.
        gt_title: Title for the ground truth labels row.
    """
    import matplotlib.pyplot as plt
    import librosa.display
    import numpy as np
    import os

    T = class_scores.shape[0]
    num_classes = class_scores.shape[1]
    sec_per_col = 0.02  # WhisperFeatureExtractor → 50 Hz
    time_axis = np.arange(T) * sec_per_col

    # Colors
    colors = plt.cm.Set2(np.linspace(0, 1, num_classes))
    color_map = {i: colors[i % len(colors)] for i in range(num_classes)}

    color_map = {0: "darkorange", 1: "cornflowerblue", 2: "gold", 3: "r"}
    map = {"m": "moan", "h": "hmm", "w": "wail"}
    unique_labels = ["m", "h", "w"]

    # 4 rows if extra labels exist, otherwise 3
    has_extra = (
        extra_onsets is not None
        and extra_offsets is not None
        and extra_labels is not None
    )
    nrows = 4 if has_extra else 3
    height_ratios = [3, 1, 0.6] + ([0.6] if has_extra else [])

    fig, axs = plt.subplots(
        nrows, 1, figsize=(12, 7 if has_extra else 6), height_ratios=height_ratios
    )
    if nrows == 3:
        ax_spec, ax_scores, ax_gt = axs
    else:
        ax_spec, ax_scores, ax_gt, ax_extra = axs

    # 1) Mel spectrogram
    # mel_spec: (n_mels, T)
    n_mels = mel_spec.shape[0]
    fmin = 0
    fmax = 8000

    # Mel bin frequencies in Hz
    mel_f = librosa.mel_frequencies(n_mels=n_mels, fmin=fmin, fmax=fmax)

    # 1) Mel spectrogram
    librosa.display.specshow(
        mel_spec,
        cmap=plt.cm.magma,
        sr=16000,
        hop_length=160,
        x_axis="time",
        y_axis="mel",  # <-- sorgt für die richtige Mel-Skala
        fmin=0,
        fmax=8000,
        ax=ax_spec,
    )
    ax_spec.set_ylabel("Frequency (Hz)")

    # 2) Scores + Ground Truth
    frame_width = sec_per_col
    for c in range(num_classes):
        ax_scores.bar(
            time_axis,
            class_scores[:, c],
            width=frame_width,
            align="edge",
            alpha=1,
            label=f"{map[ID_TO_CLUSTER[c]]}",
            color=color_map[c],
        )

    ax_scores.axhline(
        y=threshold, color="r", linestyle="--", label=f"Threshold {threshold}"
    )
    ax_scores.set_ylim(0, 1.1)
    ax_scores.set_title("WhisperFormer Scores and Labels")
    ax_scores.set_xlabel("Time (s)")
    ax_scores.set_ylabel("Score")
    ax_scores.set_xlim(0, T * sec_per_col)

    # WhisperFormer Predictions
    for onset, offset, c in zip(pred_onsets, pred_offsets, pred_classes):
        color = color_map[FIXED_CLUSTER_CODEBOOK[c]]
        ax_scores.axvspan(onset, offset, color=color, alpha=0.3)
        mid = (onset + offset) / 2

        # Calculate SNR
        sr = 16000
        start_sample = int(onset * sr)
        end_sample = int(offset * sr)
        segment_audio = y
        snr_value = compute_snr_new(segment_audio, sr, cutoff=200)
        snr_value = compute_snr_timebased(segment_audio, sr, start_sample, end_sample)

    patches = [
        mpatches.Patch(color=color_map[FIXED_CLUSTER_CODEBOOK[l]], label=str(map[l]))
        for l in unique_labels
    ]
    ax_scores.legend(handles=patches, loc="upper right")

    # Deduplicate legend entries
    handles, labels = ax_scores.get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    ax_scores.legend(by_label.values(), by_label.keys(), loc="upper right")

    # 3) Ground Truth Labels
    if has_extra:
        ax_gt.set_title(gt_title)
        ax_gt.set_xlim(0, T * sec_per_col)
        ax_gt.set_ylim(0, 1)
        ax_gt.set_xlabel("Time (s)")
        ax_gt.set_yticks([])

        # unique_labels = sorted(set(gt_classes))
        unique_labels = ["m", "h", "w"]
        label_colors = plt.cm.Paired(np.linspace(0, 1, len(unique_labels)))
        label_color_map = {lab: label_colors[i] for i, lab in enumerate(unique_labels)}
        # farben_extra = farben[np.array([np.where(np.unique(gt_labels) == lab)[0][0] for lab in extra_labels])]

        for onset, offset, lab, q in zip(
            gt_onsets, gt_offsets, gt_classes, gt_qualities
        ):
            # color = label_color_map[lab]
            color = color_map[FIXED_CLUSTER_CODEBOOK[lab]]
            ax_gt.axvspan(onset, offset, color=color, alpha=0.4)
            mid = (onset + offset) / 2
            ax_gt.text(
                mid,
                0.5,
                str(q),
                ha="center",
                va="center",
                fontsize=8,
                fontweight="bold",
                color="black",
                bbox=dict(facecolor="white", edgecolor="none", alpha=0.7, pad=0.5),
            )

        # Legend for GT labels
        patches = [
            mpatches.Patch(
                color=color_map[FIXED_CLUSTER_CODEBOOK[l]], label=str(map[l])
            )
            for l in unique_labels
        ]
        ax_gt.legend(handles=patches, loc="upper right")

    # 4) Extra labels (optional)
    if has_extra:
        ax_extra.set_title(extra_title)
        ax_extra.set_xlim(0, T * sec_per_col)
        ax_extra.set_ylim(0, 1)
        ax_extra.set_xlabel("Time (s)")
        ax_extra.set_yticks([])

        # unique_labels = sorted(set(extra_labels))
        unique_labels = ["m", "h", "w"]
        label_colors = plt.cm.Paired(np.linspace(0, 1, len(unique_labels)))
        label_color_map = {lab: label_colors[i] for i, lab in enumerate(unique_labels)}
        # farben_extra = farben[np.array([np.where(np.unique(gt_labels) == lab)[0][0] for lab in extra_labels])]

        for onset, offset, lab in zip(extra_onsets, extra_offsets, extra_labels):
            # color = label_color_map[lab]
            color = color_map[FIXED_CLUSTER_CODEBOOK[lab]]
            ax_extra.axvspan(onset, offset, color=color, alpha=0.4)
            mid = onset + offset
            """
            ax_extra.text(
                mid, 0.5, str(lab),
                ha='center',
                va='center',
                fontsize=8,
                fontweight='bold',
                color='black',
                bbox=dict(facecolor='white', edgecolor='none', alpha=0.7, pad=0.5)
            )
            """

        # Legend for extra labels
        patches = [
            mpatches.Patch(
                color=color_map[FIXED_CLUSTER_CODEBOOK[l]], label=str(map[l])
            )
            for l in unique_labels
        ]
        ax_extra.legend(handles=patches, loc="upper right")

    plt.tight_layout()

    save_filename = f"{base_name}_segment_{segment_idx:02d}_spectrogram_scores_gt.png"
    save_path = os.path.join(save_dir, save_filename)
    plt.savefig(save_path, dpi=150)
    plt.close()
    # print(f"Segment plot saved to {save_path}")


# ==================== MODEL LOADING ====================


def load_trained_whisperformer(checkpoint_path, num_classes, device, whisper_size=None):
    """Load a trained WhisperFormer model from a checkpoint.

    Architecture parameters (num_decoder_layers, num_head_layers, num_classes)
    are inferred automatically from the checkpoint keys.

    The checkpoint contains all weights (including the frozen Whisper
    encoder), so only the lightweight ``WhisperConfig`` JSON is loaded from
    disk -- the full pretrained Whisper model is **not** downloaded.  The
    Whisper size is auto-detected from the checkpoint weights
    (d_model=512 -> base, d_model=1280 -> large) unless explicitly specified.

    Args:
        checkpoint_path: Path to the ``.pt`` checkpoint file.
        num_classes: Number of output classes (overridden if detectable from checkpoint).
        device: Torch device string or object (e.g. ``"cuda"`` or ``"cpu"``).
        whisper_size: Optional explicit Whisper size (``"base"`` or ``"large"``).
            Auto-detected from the checkpoint if ``None``.

    Returns:
        Tuple of ``(model, detected_size)`` where *model* is the loaded
        ``WhisperFormer`` in eval mode and *detected_size* is ``"base"`` or
        ``"large"``.
    """
    from .whisperformer.model import (
        infer_architecture_from_state_dict,
        detect_whisper_size_from_state_dict,
    )

    # 1) Load state dict
    state_dict = torch.load(checkpoint_path, map_location=device)

    # 2) Infer architecture from checkpoint
    num_decoder_layers, num_head_layers, ckpt_num_classes = (
        infer_architecture_from_state_dict(state_dict)
    )
    if ckpt_num_classes is not None:
        num_classes = ckpt_num_classes
    print(
        f"Checkpoint: num_decoder_layers={num_decoder_layers}, num_head_layers={num_head_layers}, num_classes={num_classes}"
    )

    # 3) Determine Whisper size
    if whisper_size and whisper_size.lower() in ["base", "large"]:
        detected_size = whisper_size.lower()
    else:
        detected_size = detect_whisper_size_from_state_dict(state_dict)
        if detected_size is None:
            # Fallback: infer from checkpoint path
            path_lower = checkpoint_path.lower()
            if "large" in path_lower:
                detected_size = "large"
            else:
                detected_size = "base"

    print(f"Detected Whisper size: {detected_size}")

    # 4) Load config only (small JSON) -- no from_pretrained needed,
    #    since all weights (incl. encoder) come from the checkpoint
    _project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    _whisper_models_dir = os.path.join(_project_root, "whisper_models")
    config_path = os.path.join(_whisper_models_dir, f"whisper_{detected_size}")

    config = WhisperConfig.from_pretrained(config_path)
    whisper_model = WhisperModel(config)  # empty weights, no download
    encoder = whisper_model.encoder

    model = WhisperFormer(
        encoder,
        num_classes=num_classes,
        num_decoder_layers=num_decoder_layers,
        num_head_layers=num_head_layers,
    )

    # 4) Prefix fix: checkpoint has "encoder.X", model expects "encoder.encoder.X"
    #    due to the WhisperEncoder wrapper in the WhisperFormer model
    needs_remap = any(
        k.startswith("encoder.") and not k.startswith("encoder.encoder.")
        for k in state_dict.keys()
    )
    if needs_remap:
        new_state_dict = {}
        for k, v in state_dict.items():
            if k.startswith("encoder.") and not k.startswith("encoder.encoder."):
                new_state_dict["encoder." + k] = v
            else:
                new_state_dict[k] = v
        state_dict = new_state_dict

    # 5) Load ALL weights (encoder + decoder + heads) from the checkpoint
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()
    return model, detected_size


# ==================== INFERENCE ====================


def run_inference_new(
    model, dataloader, device, threshold, iou_threshold, metadata_list
):
    """Run inference and map each prediction to its corresponding slice.

    Iterates over the dataloader, applies the model with optional mixed-precision,
    filters predictions by *threshold*, and applies NMS per class.

    Args:
        model: Trained ``WhisperFormer`` model in eval mode.
        dataloader: DataLoader created with ``shuffle=False`` so that slice
            order matches *metadata_list*.
        device: Torch device string or object.
        threshold: Minimum score to keep a predicted interval.
        iou_threshold: IoU threshold for non-maximum suppression.
        metadata_list: List of slice metadata dicts (from
            ``slice_audios_and_labels``), each containing ``original_idx``
            and ``segment_idx``.

    Returns:
        List of dicts, one per slice::

            {
                "original_idx": int,
                "segment_idx": int,
                "preds": [{"class": c, "intervals": [[start, end, score], ...]}, ...]
            }
    """
    preds_by_slice = []
    slice_idx = 0
    model.eval()

    with torch.no_grad():
        for batch in dataloader:
            # Move tensors to device
            for k, v in batch.items():
                if isinstance(v, torch.Tensor):
                    batch[k] = v.to(device, non_blocking=True)

            # Enable autocast only on CUDA
            use_autocast = (isinstance(device, str) and device.startswith("cuda")) or (
                hasattr(device, "type") and device.type == "cuda"
            )
            autocast_ctx = (
                torch.amp.autocast(device_type="cuda", dtype=torch.float16)
                if use_autocast
                else contextlib.nullcontext()
            )

            with autocast_ctx:
                class_preds, regr_preds = model(batch["input_features"])
                class_probs = torch.sigmoid(class_preds)

            B, T, C = class_preds.shape
            for b in range(B):
                # Get matching slice from metadata_list
                meta = metadata_list[slice_idx]
                slice_idx += 1

                preds_per_class = []
                for c in range(C):
                    intervals = []
                    for t in range(T):
                        score = class_probs[b, t, c]
                        if float(score) > threshold:
                            start = t - regr_preds[b, t, 0]
                            end = t + regr_preds[b, t, 1]
                            intervals.append(torch.stack([start, end, score]))

                    if len(intervals) > 0:
                        intervals = torch.stack(intervals)
                        intervals = nms_1d_torch(intervals, iou_threshold=iou_threshold)
                        intervals = intervals.cpu().tolist()

                    else:
                        intervals = []

                    preds_per_class.append({"class": c, "intervals": intervals})

                preds_by_slice.append(
                    {
                        "original_idx": meta["original_idx"],
                        "segment_idx": meta["segment_idx"],
                        "preds": preds_per_class,
                    }
                )

    # Sanity check: number of slices must match
    assert len(preds_by_slice) == len(metadata_list), (
        f"Prediction list ({len(preds_by_slice)}) does not match metadata list ({len(metadata_list)}). "
        "Make sure DataLoader uses shuffle=False and the ordering is consistent."
    )

    return preds_by_slice


def reconstruct_predictions(preds_by_slice, total_spec_columns, ID_TO_CLUSTER):
    """Reconstruct predictions from slice-local coordinates to file-level timestamps.

    Merges per-slice predictions across all segments of each original audio
    file, converting frame indices back to seconds.

    Args:
        preds_by_slice: List of per-slice prediction dicts as returned by
            :func:`run_inference_new`.
        total_spec_columns: Number of spectrogram columns per slice (before
            the 2x downsampling in the model).
        ID_TO_CLUSTER: Mapping from integer class ID to cluster label string.

    Returns:
        Dict with keys ``"onset"``, ``"offset"``, ``"cluster"``, and
        ``"score"``, each mapping to a list of values across all files.
    """
    grouped_preds = defaultdict(list)
    for ps in preds_by_slice:
        grouped_preds[ps["original_idx"]].append(ps)

    sec_per_col = 0.02
    cols_per_segment = total_spec_columns // 2  # T entspricht total_spec_columns/2

    all_preds_final = {"onset": [], "offset": [], "cluster": [], "score": []}

    # Iterate over all original files
    for orig_idx in sorted(grouped_preds.keys()):
        segs_sorted = sorted(grouped_preds[orig_idx], key=lambda x: x["segment_idx"])
        for seg in segs_sorted:
            offset_cols = seg["segment_idx"] * cols_per_segment
            for p in seg["preds"]:
                c = p["class"]
                for start_col, end_col, score in p["intervals"]:
                    start_sec = (offset_cols + start_col) * sec_per_col
                    end_sec = (offset_cols + end_col) * sec_per_col
                    all_preds_final["onset"].append(float(start_sec))
                    all_preds_final["offset"].append(float(end_sec))
                    # Map class ID -> cluster label
                    # all_preds_final["cluster"].append(ID_TO_CLUSTER[c] if c in range(len(ID_TO_CLUSTER)) else "unknown")
                    all_preds_final["cluster"].append(ID_TO_CLUSTER.get(c, "unknown"))
                    all_preds_final["score"].append(float(score))

    return all_preds_final


# ==================== MAIN ====================

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint_path", required=True)
    parser.add_argument("--audio_folder", required=True)
    parser.add_argument("--label_folder", required=True)
    parser.add_argument("--pred_label_folder", required=True)
    parser.add_argument("--extra_label_folder", type=str, default=None)
    parser.add_argument("--output_dir", default="inference_outputs")
    parser.add_argument("--total_spec_columns", type=int, default=3000)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--num_classes", type=int, default=1)
    parser.add_argument("--threshold", type=float, default=0.35)
    parser.add_argument("--iou_threshold", type=float, default=0.4)
    parser.add_argument("--overlap_tolerance", type=float, default=0.1)
    parser.add_argument(
        "--device", default="cuda" if torch.cuda.is_available() else "cpu"
    )
    parser.add_argument("--low_quality_value", type=float, default=0.5)
    parser.add_argument("--value_q2", type=float, default=1)
    parser.add_argument("--centerframe_size", type=float, default=0.6)
    parser.add_argument("--allowed_qualities", default=[1, 2, 3])
    parser.add_argument(
        "--whisper_size",
        type=str,
        default=None,
        choices=["base", "large"],
        help="Whisper size (base/large). Auto-detected from checkpoint if not specified.",
    )
    args = parser.parse_args()

    # === Create timestamped output subdirectory ===
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    save_dir = os.path.join(args.output_dir, timestamp)
    os.makedirs(save_dir, exist_ok=True)

    # === Save run arguments ===
    args_path = os.path.join(save_dir, "run_arguments.json")
    with open(args_path, "w") as f:
        json.dump(vars(args), f, indent=2)
    print(f"Arguments saved to: {args_path}")

    # os.makedirs(args.output_dir, exist_ok=True)

    audio_paths, label_paths = get_audio_and_label_paths_from_folders(
        args.audio_folder, args.label_folder
    )
    cluster_codebook = FIXED_CLUSTER_CODEBOOK
    id_to_cluster = ID_TO_CLUSTER

    model, detected_size = load_trained_whisperformer(
        args.checkpoint_path,
        args.num_classes,
        args.device,
        whisper_size=args.whisper_size,
    )
    print(f"Model loaded (Whisper {detected_size})")

    # Feature extractor is independent of model size (same mel parameters)
    _project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    _whisper_models_dir = os.path.join(_project_root, "whisper_models")
    feature_extractor = WhisperFeatureExtractor.from_pretrained(
        os.path.join(_whisper_models_dir, "whisper_base"), local_files_only=True
    )

    all_labels = {"onset": [], "offset": [], "cluster": [], "quality": []}
    all_preds_final = {"onset": [], "offset": [], "cluster": [], "score": []}

    for audio_path, label_path in zip(audio_paths, label_paths):
        # print(f"\n===== Processing {os.path.basename(audio_path)} =====")
        audio_list, label_list = load_data(
            [audio_path], [label_path], cluster_codebook=cluster_codebook, n_threads=1
        )
        audio_list, label_list, metadata_list = slice_audios_and_labels(
            audio_list, label_list, args.total_spec_columns
        )

        dataset = WhisperFormerDatasetQuality(
            audio_list,
            label_list,
            args.total_spec_columns,
            feature_extractor,
            args.num_classes,
            args.low_quality_value,
            args.value_q2,
            args.centerframe_size,
        )
        dataloader = DataLoader(
            dataset,
            batch_size=args.batch_size,
            shuffle=False,
            collate_fn=collate_fn,
            drop_last=False,
        )

        preds_by_slice = run_inference_new(
            model=model,
            dataloader=dataloader,  # muss mit shuffle=False erstellt sein
            device=args.device,
            threshold=args.threshold,
            iou_threshold=args.iou_threshold,
            metadata_list=metadata_list,  # kommt aus slice_audios_and_labels
        )
        # Load audio
        y, sr = librosa.load(audio_path, sr=16000)

        # Load labels
        with open(label_path, "r") as f:
            labels = json.load(f)

        clusters = labels["cluster"]
        labels["cluster"] = [ID_TO_CLUSTER[FIXED_CLUSTER_CODEBOOK[c]] for c in clusters]

        # Add quality classes
        if "quality" in labels:
            quality_list = labels["quality"]
        else:
            quality_list = ["unknown"] * len(labels["onset"])

        # === Load WhisperFormer prediction labels for this audio file ===
        pred_labels_data = None
        if args.pred_label_folder is not None:
            base_name = os.path.splitext(os.path.basename(audio_path))[0]
            pattern_jsonr = os.path.join(args.pred_label_folder, f"{base_name}*.jsonr")
            pattern_json = os.path.join(args.pred_label_folder, f"{base_name}*.json")
            matching_files = sorted(glob.glob(pattern_json) + glob.glob(pattern_jsonr))

            if len(matching_files) > 0:
                pred_path = matching_files[0]  # Take the first matching file
                with open(pred_path, "r") as f:
                    pred_labels_data = json.load(f)
                print(
                    f"Loaded WhisperFormer predictions for {base_name} from {pred_path}"
                )
            else:
                print(f"No matching WhisperFormer predictions found for {base_name}")

        # === Load extra labels for this audio file (optional) ===
        extra_labels_data = None
        if args.extra_label_folder is not None:
            base_name = os.path.splitext(os.path.basename(audio_path))[0]
            pattern_jsonr = os.path.join(args.extra_label_folder, f"{base_name}*.jsonr")
            pattern_json = os.path.join(args.extra_label_folder, f"{base_name}*.json")
            matching_files = sorted(glob.glob(pattern_json) + glob.glob(pattern_jsonr))

            if len(matching_files) > 0:
                extra_path = matching_files[0]  # Take the first matching file
                with open(extra_path, "r") as f:
                    extra_labels_data = json.load(f)
                print(f"Loaded extra labels for {base_name} from {extra_path}")
            else:
                print(f"No matching extra labels found for {base_name}")

        # === Visualize first 3 segments with ground truth ===
        if len(dataset) > 0:
            base_name = os.path.splitext(os.path.basename(audio_path))[0]
            # print(f"Visualizing first 3 segments of {base_name} with ground truth ...")

            # Load labels for the entire file
            gt_onsets = np.array(labels["onset"])
            gt_offsets = np.array(labels["offset"])
            gt_classes = np.array(labels["cluster"])
            gt_qualities = np.array(labels["quality"])

            # Duration per segment (seconds)
            seg_dur = (
                args.total_spec_columns / 2
            ) * 0.02  # Whisper T/2 frames -> 20 ms/frame
            for i in range(min(3, len(dataset))):
                seg_start = i * seg_dur
                seg_end = (i + 1) * seg_dur

                # Crop audio to segment
                y_part = y[int(seg_start * sr) : int(seg_end * sr)]

                # Select ground truth events within this segment
                in_seg = (gt_onsets < seg_end) & (gt_offsets > seg_start)
                gt_onsets_seg = gt_onsets[in_seg] - seg_start
                gt_offsets_seg = gt_offsets[in_seg] - seg_start
                gt_classes_seg = gt_classes[in_seg]
                gt_qualities_seg = gt_qualities[in_seg]

                if not pred_labels_data:
                    pred_onsets = np.array([])
                    pred_offsets = np.array([])
                    pred_labels = np.array([])
                    pred_scores = np.array([])
                else:
                    pred_onsets = np.array(pred_labels_data.get("onset", []))
                    pred_offsets = np.array(pred_labels_data.get("offset", []))
                    pred_labels = np.array(pred_labels_data.get("cluster", []))
                    pred_scores = np.array(pred_labels_data.get("score", []))

                # Segment filter
                in_seg = (pred_onsets < seg_end) & (pred_offsets > seg_start)

                # Score filter
                score_mask = pred_scores > args.threshold

                # Combined filter
                mask = in_seg & score_mask

                pred_onsets_seg = pred_onsets[mask] - seg_start
                pred_offsets_seg = pred_offsets[mask] - seg_start
                pred_labels_seg = pred_labels[mask]
                pred_scores_seg = pred_scores[mask]
                """
                #etra labels im segment auswählen
                if extra_labels_data is not None:
                    extra_onsets = np.array(extra_labels_data.get("onset", []))
                    extra_offsets = np.array(extra_labels_data.get("offset", []))
                    extra_labels = np.array(extra_labels_data.get("cluster", []))

                    in_seg = (extra_onsets < seg_end) & (extra_offsets > seg_start)
                    extra_onsets_seg = extra_onsets[in_seg] - seg_start
                    extra_offsets_seg = extra_offsets[in_seg] - seg_start
                    extra_labels_seg = extra_labels[in_seg]
                """

                if not extra_labels_data:
                    extra_onsets = np.array([])
                    extra_offsets = np.array([])
                    extra_labels = np.array([])
                    # pred_scores = np.array([])
                else:
                    extra_onsets = np.array(extra_labels_data.get("onset", []))
                    extra_offsets = np.array(extra_labels_data.get("offset", []))
                    extra_labels = np.array(extra_labels_data.get("cluster", []))
                    # _scores  = np.array(extra_labels_data.get("score", []))

                in_seg = (extra_onsets < seg_end) & (extra_offsets > seg_start)
                extra_onsets_seg = extra_onsets[in_seg] - seg_start
                extra_offsets_seg = extra_offsets[in_seg] - seg_start
                extra_labels_seg = extra_labels[in_seg]

                # Get input features (mel spectrogram)
                features = (
                    dataset[i]["input_features"].squeeze(0).cpu().numpy()
                )  # (80, 3000)
                mel_spec = features

                # Model scores
                with torch.no_grad():
                    x = dataset[i]["input_features"].unsqueeze(0).to(args.device)
                    class_preds, _ = model(x)
                    class_scores = torch.sigmoid(class_preds).squeeze(0).cpu().numpy()

                plot_spectrogram_and_scores(
                    mel_spec=mel_spec,
                    class_scores=class_scores,
                    gt_onsets=gt_onsets_seg,
                    gt_offsets=gt_offsets_seg,
                    gt_classes=gt_classes_seg,
                    gt_qualities=gt_qualities_seg,
                    pred_onsets=pred_onsets_seg,
                    pred_offsets=pred_offsets_seg,
                    pred_classes=pred_labels_seg,
                    segment_idx=i,
                    save_dir=save_dir,
                    base_name=base_name,
                    y=y_part,
                    threshold=args.threshold,
                    ID_TO_CLUSTER=ID_TO_CLUSTER,
                    extra_onsets=extra_onsets_seg,
                    extra_offsets=extra_offsets_seg,
                    extra_labels=extra_labels_seg,
                )
