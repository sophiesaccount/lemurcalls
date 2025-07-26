import torch
import numpy as np
from torch.utils.data import Dataset
import librosa
import json


class ActionFormerDataset(Dataset):
    """Dataset for ActionFormer model - converts audio to frame-wise classification"""
    
    def __init__(self, audio_list, label_list, max_length=100, total_spec_columns=1000):
        self.audio_list = audio_list
        self.label_list = label_list
        self.max_length = max_length
        self.total_spec_columns = total_spec_columns
        
        # Create feature extractor once
        from transformers import WhisperFeatureExtractor
        self.feature_extractor = WhisperFeatureExtractor.from_pretrained("openai/whisper-small")
        
        # Automatically calculate num_classes from labels
        self.cluster_mapping = self.create_cluster_mapping()
        self.num_classes = len(self.cluster_mapping)
        print(f"Found {self.num_classes} different clusters: {list(self.cluster_mapping.keys())}")
        
    def create_cluster_mapping(self):
        """Create mapping from cluster names to class IDs by analyzing all labels"""
        unique_clusters = set()
        
        # Collect all unique cluster names from all labels
        for label in self.label_list:
            if "segments" in label:
                # New format with segments list
                segments = label.get("segments", [])
                for segment in segments:
                    cluster = segment.get("cluster", "unknown")
                    unique_clusters.add(cluster)
            elif "cluster" in label:
                # Old format with arrays (from load_data)
                clusters = label.get("cluster", [])
                for cluster in clusters:
                    unique_clusters.add(cluster)
            else:
                print(f"Warning: Unknown label format. Label keys: {list(label.keys())}")
        
        print(f"Found unique clusters in labels: {sorted(list(unique_clusters))}")
        
        # Create mapping (background = 0, others = 1, 2, 3, ...)
        cluster_mapping = {"background": 0}  # Background is always class 0
        
        # Sort clusters for consistent ordering
        sorted_clusters = sorted(list(unique_clusters))
        for i, cluster in enumerate(sorted_clusters):
            if cluster != "background":
                cluster_mapping[cluster] = i + 1  # Start from 1, since 0 is background
        
        print(f"Created cluster mapping: {cluster_mapping}")
        return cluster_mapping
        
    def __len__(self):
        return len(self.audio_list)
    
    def __getitem__(self, idx):
        audio = self.audio_list[idx]
        label = self.label_list[idx]
        
        # Extract Whisper features (same as original)
        input_features = self.extract_whisper_features(audio, label["sr"])
        
        # Convert label segments to frame-wise labels
        frame_labels = self.segments_to_frame_labels(label, input_features.shape[0])
        
        return {
            "input_features": input_features,
            "labels": frame_labels
        }
    
    def extract_whisper_features(self, audio, sr):
        """Extract Whisper features from audio"""
        # Ensure audio is numpy array
        if isinstance(audio, torch.Tensor):
            audio = audio.numpy()
        
        # Pad or truncate audio to match total_spec_columns
        target_length = int(self.total_spec_columns * 0.02 * sr)  # 20ms per frame
        if len(audio) > target_length:
            audio = audio[:target_length]
        else:
            audio = np.pad(audio, (0, target_length - len(audio)))
        
        # Extract features without padding (we handle it manually)
        features = self.feature_extractor(
            audio, 
            sampling_rate=sr, 
            return_tensors="pt",
            padding=False  # No padding, we handle it ourselves
        )
        input_features = features.input_features.squeeze(0)  # Remove batch dimension
        
        # Ensure we have the right shape
        if len(input_features.shape) == 1:
            input_features = input_features.unsqueeze(0)
        
        # Truncate to max_length
        if input_features.shape[0] > self.max_length:
            input_features = input_features[:self.max_length]
        
        return input_features
    
    def segments_to_frame_labels(self, label, num_frames):
        """Convert segment labels to frame-wise labels"""
        # Initialize all frames as background (class 0)
        frame_labels = np.zeros(num_frames, dtype=np.int64)
        
        # Handle different label formats
        if "segments" in label:
            # New format with segments list
            segments = label.get("segments", [])
            for segment in segments:
                onset = segment.get("onset", 0)
                offset = segment.get("offset", 0)
                cluster = segment.get("cluster", "unknown")
                
                # Convert time to frame indices
                onset_frame = int(onset / self.frame_duration)
                offset_frame = int(offset / self.frame_duration)
                
                # Map cluster to class ID
                class_id = self.cluster_to_class_id(cluster)
                
                # Set frames in this segment to the class ID
                onset_frame = max(0, min(onset_frame, num_frames - 1))
                offset_frame = max(0, min(offset_frame, num_frames - 1))
                
                frame_labels[onset_frame:offset_frame] = class_id
                
        elif "onset" in label and "offset" in label and "cluster" in label:
            # Old format with arrays (from load_data)
            onsets = label.get("onset", [])
            offsets = label.get("offset", [])
            clusters = label.get("cluster", [])
            
            # Convert time to frame indices
            frame_duration = 0.02  # 20ms per frame (Whisper default)
            
            for onset, offset, cluster in zip(onsets, offsets, clusters):
                onset_frame = int(onset / frame_duration)
                offset_frame = int(offset / frame_duration)
                
                # Map cluster to class ID
                class_id = self.cluster_to_class_id(cluster)
                
                # Set frames in this segment to the class ID
                onset_frame = max(0, min(onset_frame, num_frames - 1))
                offset_frame = max(0, min(offset_frame, num_frames - 1))
                
                frame_labels[onset_frame:offset_frame] = class_id
        else:
            print(f"Warning: Unknown label format for sample. Label keys: {list(label.keys())}")
        
        return torch.tensor(frame_labels, dtype=torch.long)
    
    def cluster_to_class_id(self, cluster):
        """Map cluster names to class IDs using the automatically created mapping"""
        if cluster not in self.cluster_mapping:
            print(f"Warning: Cluster '{cluster}' not found in mapping {self.cluster_mapping}. Using background class 0.")
            return 0  # Default to background (0)
        return self.cluster_mapping[cluster] 