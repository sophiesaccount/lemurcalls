import torch
import numpy as np
from torch.utils.data import Dataset
import librosa
import json
import torchaudio
from util.common import is_scheduled_job
from utils import RATIO_DECODING_TIME_STEP_TO_SPEC_TIME_STEP


class WhisperFormerDatasetQuality(Dataset):
    def __init__(self, audio_list, label_list, total_spec_columns, feature_extractor, num_classes, low_quality_value, value_q2, centerframe_size):
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
        audio = self.audio_list[idx]
        label = self.label_list[idx]


        # 1) In 1D np.float32 umwandeln
        if isinstance(audio, torch.Tensor):
            audio = audio.detach().cpu().numpy()
        audio = np.asarray(audio).astype(np.float32).reshape(-1)

        # 2) Resampling auf 16 kHz (falls nötig)
        target_sr = 16000
        orig_sr = int(label.get("sr", target_sr))
        if orig_sr != target_sr:
            # librosa erwartet float32 [-1,1]
            audio = librosa.resample(audio, orig_sr=orig_sr, target_sr=target_sr)

        # 3) Feste Clip-Länge in Samples (z. B. 3000 Frames * 10 ms * 16000 Hz = 480000)
        spec_time_step = 0.01  # 10 ms
        num_samples_in_clip = int(round(self.total_spec_columns * spec_time_step * target_sr))

        clip_start = 0
        audio = audio[clip_start : clip_start + num_samples_in_clip]


        # 4) Pad oder kürzen auf exakt num_samples_in_clip
        if len(audio) < num_samples_in_clip:
            pad_len = num_samples_in_clip - len(audio)
            audio = np.pad(audio, (0, pad_len), mode="constant")
        elif len(audio) > num_samples_in_clip:
            audio = audio[:num_samples_in_clip]


        # 5) Whisper-Features mit fixer Frame-Länge erzeugen
        input_features = self.feature_extractor(
            audio,
            sampling_rate=target_sr,
            return_tensors="pt",
            padding="max_length",
            truncation=True,
            max_length=num_samples_in_clip, 
        ).input_features  

        input_features = input_features.squeeze(0) # -> (80, 3000)

        sr = 16000
        spec_time_step = 0.01
        total_spec_columns=3000
        n_fft = 400

        actual_clip_duration = len(audio) / sr
        start_time = clip_start / sr #0
        end_time = start_time + actual_clip_duration
      
 
        #label finden die mit clip überschneiden
        intersected_indices = np.logical_and(label["onset"] < end_time, label["offset"] > start_time) 

        onset_in_clip = np.maximum(label["onset"][intersected_indices], start_time) 
        offset_in_clip = np.minimum(label["offset"][intersected_indices], end_time) 
        cluster_id_in_clip = label["cluster_id"][intersected_indices]
        quality_in_clip = [label["quality"][idx] for idx in np.argwhere(intersected_indices)[:, 0]]
        #quality_array = label.get("quality", None)
        #if quality_array is None:
        #    label['quality'] = ["unknown"] * len(np.argwhere(intersected_indices)[:, 0])
        #else:
        #    label['quality'] = [quality_array[idx] for idx in np.argwhere(intersected_indices)[:, 0]]

        sequence_length = self.total_spec_columns #3000
        reduced_sequence_length = sequence_length // 2 #1500

        reduced_spec_time_step = spec_time_step * 2  #0.02

        # Frame-basierte Label-Matrix (mit Gaussians)
        frame_labels = np.zeros((reduced_sequence_length, self.num_classes), dtype=np.float32) #(1500, C)

        # Segments
        segments = np.zeros((reduced_sequence_length, 2), dtype=np.float32) #(1500, 2)

        offset_mask_size = 2
        offset_mask = np.zeros(reduced_sequence_length, dtype=bool)

        for onset, offset, quality, cluster_id in zip(onset_in_clip, offset_in_clip, quality_in_clip, cluster_id_in_clip):
            if cluster_id < 0 or cluster_id >= self.num_classes:
                continue
            

            # Sekunden -> Spektrogramm-Frames
            onset_col = int(np.floor(onset / reduced_spec_time_step))
            offset_col = min(int(np.ceil(offset / reduced_spec_time_step)), reduced_sequence_length-1)


            # Mittelpunkt als ganzzahliger Frame
            center_frame = (onset_col + offset_col) // 2
            frame_len = offset_col-onset_col

            offset_mask[center_frame - int(self.centerframe_size*(frame_len/2)) : center_frame + int(self.centerframe_size*(frame_len/2))] = True
            offset_mask = offset_mask.reshape(-1,1)

            # Reduzierte Frames
            #onset_red = onset_col // 2
            #offset_red = offset_col // 2
            #center_red = center_frame // 2

            # Frames only inside the Labels
            label_frames = np.arange(onset_col, offset_col+1)
            sigma = max(offset_col - onset_col, 1e-6)

            gaussian = np.exp(-0.5 * ((label_frames - center_frame) / sigma) ** 2)

            frame_update = np.zeros(reduced_sequence_length, dtype=np.float32)
            if str(quality) != "3" and str(quality) != "2":
                frame_update[label_frames] = 1
            elif str(quality) == "2":
                frame_update[label_frames] = self.value_q2
            else: 
                frame_update[label_frames] = self.low_quality_value
            # Frame-Labels aktualisieren
            frame_labels[:, cluster_id] = np.maximum(
                frame_labels[:, cluster_id], frame_update
            )

            frame_labels= frame_labels * offset_mask

            # Segments für diese Klasse eintragen
            for t in label_frames:
                segments[t, 0] = t - onset_col     # Abstand zum Onset
                segments[t, 1] = offset_col - t    # Abstand zum Offset

            #segments= segments * offset_mask

        output = {
            "input_features": input_features,   # (B, F, T)
            "clusters": torch.tensor(frame_labels, dtype=torch.float32),          # (B, T/2, C)
            "segments": torch.tensor(segments, dtype=torch.float32),              # (B, T/2, 2)
            "raw_labels": label
        }

        return output



