import torch
import numpy as np
from torch.utils.data import Dataset
import librosa
import json

from audio_utils import WhisperSegFeatureExtractor
from util.common import is_scheduled_job
from utils import RATIO_DECODING_TIME_STEP_TO_SPEC_TIME_STEP


class WhisperFormerDataset(Dataset):
    def __init__(self, audio_list, label_list, tokenizer, max_length, total_spec_columns, num_classes=2):
        self.audio_list = audio_list
        self.label_list = label_list
        self.feature_extractor_bank = self.get_feature_extractor_bank(label_list)
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.total_spec_columns = total_spec_columns
        self.num_classes = num_classes

    def get_feature_extractor_bank(self, label_list):
        feature_extractor_bank = {}
        for label in label_list:
            key = "%s-%s-%s" % (str(label["sr"]), str(label["spec_time_step"]), str(label["min_frequency"]))
            if key not in feature_extractor_bank:
                feature_extractor_bank[key] = WhisperSegFeatureExtractor(
                    label["sr"], label["spec_time_step"], label["min_frequency"]
                )
        return feature_extractor_bank

    def __len__(self):
        return len(self.audio_list)

    def __getitem__(self, idx):
        audio = self.audio_list[idx]
        label = self.label_list[idx]

        sr = label["sr"]
        spec_time_step = label["spec_time_step"]
        min_frequency = label["min_frequency"]
        feature_extractor = self.feature_extractor_bank[
            "%s-%s-%s" % (str(sr), str(spec_time_step), str(min_frequency))
        ]

        num_samples_in_clip = int(np.round(self.total_spec_columns * spec_time_step * sr))
        clip_start = np.random.choice(min(num_samples_in_clip + 1, len(audio) - feature_extractor.n_fft + 1))
        audio_clip = audio[clip_start : clip_start + num_samples_in_clip]

        actual_clip_duration = len(audio_clip) / sr
        start_time = clip_start / sr
        end_time = start_time + actual_clip_duration

        intersected_indices = np.logical_and(label["onset"] < end_time, label["offset"] > start_time)

        onset_in_clip = np.maximum(label["onset"][intersected_indices], start_time) - start_time
        offset_in_clip = np.minimum(label["offset"][intersected_indices], end_time) - start_time
        cluster_id_in_clip = label["cluster_id"][intersected_indices]

        audio_clip = np.concatenate(
            [audio_clip, np.zeros(num_samples_in_clip - len(audio_clip))], axis=0
        ).astype(np.float32)

        input_features = feature_extractor(
            audio_clip, sampling_rate=sr, padding="do_not_pad"
        )["input_features"][0]

        sequence_length = self.total_spec_columns
        reduced_sequence_length = sequence_length // 2

        # Frame-basierte Label-Matrix (mit Gaussians)
        frame_labels = np.zeros((reduced_sequence_length, self.num_classes), dtype=np.float32)

        # Segments: für jede Klasse eigene Spalte [Δonset, Δoffset]
        segments = np.zeros((reduced_sequence_length, self.num_classes, 2), dtype=np.float32)

        for onset, offset, cluster_id in zip(onset_in_clip, offset_in_clip, cluster_id_in_clip):
            if cluster_id < 0 or cluster_id >= self.num_classes:
                continue

            # Sekunden -> Spektrogramm-Frames
            onset_col = int(np.floor(onset / spec_time_step))
            offset_col = int(np.ceil(offset / spec_time_step))

            # Clamping
            onset_col = max(0, min(reduced_sequence_length - 1, onset_col))
            offset_col = max(0, min(reduced_sequence_length - 1, offset_col))

            # Mittelpunkt als ganzzahliger Frame
            center_frame = (onset_col + offset_col) // 2

            # Reduzierte Frames
            onset_red = onset_col // 2
            offset_red = offset_col // 2
            center_red = center_frame // 2

            # Frames only inside the Labels
            label_frames = np.arange(onset_red, offset_red + 1)
            sigma = max(offset_red - onset_red, 1e-6)

            gaussian = np.exp(-0.5 * ((label_frames - center_red) / sigma) ** 2)

            frame_update = np.zeros(reduced_sequence_length, dtype=np.float32)
            frame_update[label_frames] = gaussian

            # ToDO: CenterSampling für Training (bzw wird probably durch Gaussian ersetzt)!!
            # Frame-Labels aktualisieren
            frame_labels[:, cluster_id] = np.maximum(
                frame_labels[:, cluster_id], frame_update
            )

            # Segments für diese Klasse eintragen
            for t in label_frames:
                segments[t, cluster_id, 0] = t - onset_red     # Abstand zum Onset
                segments[t, cluster_id, 1] = offset_red - t    # Abstand zum Offset

        # Maske: wo hat Cluster > 0 binary
        mask = (frame_labels > 0).astype(np.float32)  # (T/2, C)

        output = {
            "input_features": torch.tensor(input_features, dtype=torch.float32),  # (F, T)
            "clusters": torch.tensor(frame_labels, dtype=torch.float32),          # (T/2, C)
            "segments": torch.tensor(segments, dtype=torch.float32),              # (T/2, C, 2)
            "mask": torch.tensor(mask, dtype=torch.float32),                      # (T/2, C)
        }

        return output