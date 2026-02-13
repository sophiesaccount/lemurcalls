"""Tests for WhisperFormerDatasetQuality."""

import numpy as np
import pytest
import torch
from transformers import WhisperFeatureExtractor

from lemurcalls.whisperformer.dataset import WhisperFormerDatasetQuality


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_synthetic_data(duration_sec=30.0, sr=16000, n_events=3, num_classes=3):
    """Create a synthetic audio + label pair for dataset testing."""
    n_samples = int(duration_sec * sr)
    audio = np.random.randn(n_samples).astype(np.float32) * 0.1

    gap = duration_sec / (n_events + 1)
    onsets = np.array([gap * (i + 1) - 0.1 for i in range(n_events)])
    offsets = onsets + 0.3
    cluster_ids = np.array([i % num_classes for i in range(n_events)])
    qualities = [1] * n_events

    label = {
        "sr": sr,
        "spec_time_step": 0.01,
        "min_frequency": 0,
        "onset": onsets,
        "offset": offsets,
        "cluster_id": cluster_ids,
        "cluster": [str(i) for i in cluster_ids],
        "quality": qualities,
    }
    return audio, label


@pytest.fixture
def feature_extractor():
    """Return a minimal WhisperFeatureExtractor (does not need model weights)."""
    return WhisperFeatureExtractor(
        feature_size=80,
        sampling_rate=16000,
        hop_length=160,
        chunk_length=30,
        n_fft=400,
    )


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestWhisperFormerDatasetQuality:

    def test_output_keys(self, feature_extractor):
        audio, label = _make_synthetic_data()
        ds = WhisperFormerDatasetQuality(
            [audio], [label], 3000, feature_extractor, 3, 0.0, 0.5, 0.5
        )
        sample = ds[0]
        assert "input_features" in sample
        assert "clusters" in sample
        assert "segments" in sample
        assert "raw_labels" in sample

    def test_output_shapes(self, feature_extractor):
        """Check that output tensors have the expected shapes."""
        num_classes = 3
        total_spec_columns = 3000
        audio, label = _make_synthetic_data(num_classes=num_classes)

        ds = WhisperFormerDatasetQuality(
            [audio], [label], total_spec_columns, feature_extractor,
            num_classes, 0.0, 0.5, 0.5
        )
        sample = ds[0]

        T_reduced = total_spec_columns // 2  # 1500
        assert sample["input_features"].shape == (80, total_spec_columns)
        assert sample["clusters"].shape == (T_reduced, num_classes)
        assert sample["segments"].shape == (T_reduced, 2)

    def test_output_dtypes(self, feature_extractor):
        audio, label = _make_synthetic_data()
        ds = WhisperFormerDatasetQuality(
            [audio], [label], 3000, feature_extractor, 3, 0.0, 0.5, 0.5
        )
        sample = ds[0]
        assert sample["clusters"].dtype == torch.float32
        assert sample["segments"].dtype == torch.float32

    def test_quality_weighting(self, feature_extractor):
        """Low-quality labels should get reduced weight, not 1.0."""
        audio, label = _make_synthetic_data(n_events=1, num_classes=3)
        label["quality"] = [3]  # quality 3 = low quality

        low_q_val = 0.1
        ds = WhisperFormerDatasetQuality(
            [audio], [label], 3000, feature_extractor, 3, low_q_val, 0.5, 0.5
        )
        sample = ds[0]
        clusters = sample["clusters"].numpy()

        max_val = clusters.max()
        if max_val > 0:
            assert max_val <= low_q_val + 1e-6, (
                f"Quality-3 label should have weight <= {low_q_val}, got {max_val}"
            )

    def test_no_events_returns_zeros(self, feature_extractor):
        """A clip with no events should produce all-zero clusters and segments."""
        audio = np.random.randn(480000).astype(np.float32) * 0.1
        label = {
            "sr": 16000, "spec_time_step": 0.01, "min_frequency": 0,
            "onset": np.array([]),
            "offset": np.array([]),
            "cluster_id": np.array([], dtype=int),
            "cluster": [],
            "quality": [],
        }

        ds = WhisperFormerDatasetQuality(
            [audio], [label], 3000, feature_extractor, 3, 0.0, 0.5, 0.5
        )
        sample = ds[0]

        assert torch.all(sample["clusters"] == 0)
        assert torch.all(sample["segments"] == 0)

    def test_len(self, feature_extractor):
        audio, label = _make_synthetic_data()
        ds = WhisperFormerDatasetQuality(
            [audio], [label], 3000, feature_extractor, 3, 0.0, 0.5, 0.5
        )
        assert len(ds) == 1
