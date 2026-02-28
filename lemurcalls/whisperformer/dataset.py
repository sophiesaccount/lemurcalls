import torch
import numpy as np
from torch.utils.data import Dataset
import librosa


class WhisperFormerDatasetQuality(Dataset):
    """Dataset that yields Whisper input features and quality-weighted frame labels for WhisperFormer."""

    def __init__(
        self,
        audio_list,
        label_list,
        total_spec_columns,
        feature_extractor,
        num_classes,
        low_quality_value,
        value_q2,
        centerframe_size,
    ):
        self.audio_list = audio_list
        self.label_list = label_list
        self.total_spec_columns = total_spec_columns
        self.num_classes = num_classes
        self.feature_extractor = feature_extractor
        self.low_quality_value = low_quality_value
        self.value_q2 = value_q2
        self.centerframe_size = centerframe_size

    def __len__(self):
        return len(self.audio_list)

    def __getitem__(self, idx):
        """Returns a single sample: input_features, frame labels (clusters), segments, raw_labels."""
        audio = self.audio_list[idx]
        label = self.label_list[idx]

        # Convert to 1D np.float32
        if isinstance(audio, torch.Tensor):
            audio = audio.detach().cpu().numpy()
        audio = np.asarray(audio).astype(np.float32).reshape(-1)

        # Resample to 16 kHz if needed
        target_sr = 16000
        orig_sr = int(label.get("sr", target_sr))
        if orig_sr != target_sr:
            audio = librosa.resample(audio, orig_sr=orig_sr, target_sr=target_sr)

        # Fixed clip length in samples (e.g. 3000 frames * 10 ms * 16000 Hz = 480000)
        spec_time_step = 0.01  # 10 ms
        num_samples_in_clip = int(
            round(self.total_spec_columns * spec_time_step * target_sr)
        )

        clip_start = 0
        audio = audio[clip_start : clip_start + num_samples_in_clip]

        # Pad or truncate to exactly num_samples_in_clip
        if len(audio) < num_samples_in_clip:
            pad_len = num_samples_in_clip - len(audio)
            audio = np.pad(audio, (0, pad_len), mode="constant")
        elif len(audio) > num_samples_in_clip:
            audio = audio[:num_samples_in_clip]

        # Compute Whisper features with fixed frame length
        input_features = self.feature_extractor(
            audio,
            sampling_rate=target_sr,
            return_tensors="pt",
            padding="max_length",
            truncation=True,
            max_length=num_samples_in_clip,
        ).input_features

        input_features = input_features.squeeze(0)

        sr = 16000
        spec_time_step = 0.01
        total_spec_columns = 3000
        n_fft = 400

        actual_clip_duration = len(audio) / sr
        start_time = clip_start / sr  # 0
        end_time = start_time + actual_clip_duration

        # Find labels that overlap with this clip
        intersected_indices = np.logical_and(
            label["onset"] < end_time, label["offset"] > start_time
        )

        onset_in_clip = np.maximum(label["onset"][intersected_indices], start_time)
        offset_in_clip = np.minimum(label["offset"][intersected_indices], end_time)
        cluster_id_in_clip = label["cluster_id"][intersected_indices]
        quality_in_clip = [
            label["quality"][idx] for idx in np.argwhere(intersected_indices)[:, 0]
        ]
        # quality_array = label.get("quality", None)
        # if quality_array is None:
        #    label['quality'] = ["unknown"] * len(np.argwhere(intersected_indices)[:, 0])
        # else:
        #    label['quality'] = [quality_array[idx] for idx in np.argwhere(intersected_indices)[:, 0]]

        sequence_length = self.total_spec_columns  # 3000
        reduced_sequence_length = sequence_length // 2  # 1500

        reduced_spec_time_step = spec_time_step * 2

        # Frame-based label matrix (with Gaussians)
        frame_labels = np.zeros(
            (reduced_sequence_length, self.num_classes), dtype=np.float32
        )

        segments = np.zeros((reduced_sequence_length, 2), dtype=np.float32)

        offset_mask_size = 2
        offset_mask = np.zeros(reduced_sequence_length, dtype=bool)

        for onset, offset, quality, cluster_id in zip(
            onset_in_clip, offset_in_clip, quality_in_clip, cluster_id_in_clip
        ):
            if cluster_id < 0 or cluster_id >= self.num_classes:
                continue

            # Seconds -> spectrogram frame indices
            onset_col = int(np.floor(onset / reduced_spec_time_step))
            offset_col = min(
                int(np.ceil(offset / reduced_spec_time_step)),
                reduced_sequence_length - 1,
            )

            center_frame = (onset_col + offset_col) // 2
            frame_len = offset_col - onset_col

            offset_mask[
                center_frame
                - int(self.centerframe_size * (frame_len / 2)) : center_frame
                + int(self.centerframe_size * (frame_len / 2))
            ] = True
            offset_mask = offset_mask.reshape(-1, 1)

            label_frames = np.arange(onset_col, offset_col + 1)
            sigma = max(offset_col - onset_col, 1e-6)

            gaussian = np.exp(-0.5 * ((label_frames - center_frame) / sigma) ** 2)

            frame_update = np.zeros(reduced_sequence_length, dtype=np.float32)
            if str(quality) != "3" and str(quality) != "2":
                frame_update[label_frames] = 1
            elif str(quality) == "2":
                frame_update[label_frames] = self.value_q2
            else:
                frame_update[label_frames] = self.low_quality_value
            frame_labels[:, cluster_id] = np.maximum(
                frame_labels[:, cluster_id], frame_update
            )

            frame_labels = frame_labels * offset_mask

            for t in label_frames:
                segments[t, 0] = t - onset_col
                segments[t, 1] = offset_col - t

            # segments= segments * offset_mask

        output = {
            "input_features": input_features,
            "clusters": torch.tensor(frame_labels, dtype=torch.float32),
            "segments": torch.tensor(segments, dtype=torch.float32),
            "raw_labels": label,
        }

        return output
