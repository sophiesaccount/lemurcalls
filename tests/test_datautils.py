"""Tests for the data loading and slicing pipeline."""

import numpy as np
import pytest

from lemurcalls.datautils import (
    FIXED_CLUSTER_CODEBOOK,
    ID_TO_CLUSTER,
    slice_audios_and_labels,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_audio_and_label(duration_sec=5.0, sr=16000, n_events=3):
    """Create a synthetic audio array and a matching label dict."""
    n_samples = int(duration_sec * sr)
    audio = np.random.randn(n_samples).astype(np.float32) * 0.1

    # Space events evenly across the clip
    gap = duration_sec / (n_events + 1)
    onsets = np.array([gap * (i + 1) - 0.1 for i in range(n_events)])
    offsets = onsets + 0.2  # each event is 0.2 s long
    cluster_keys = list(FIXED_CLUSTER_CODEBOOK.keys())
    clusters = [cluster_keys[i % len(cluster_keys)] for i in range(n_events)]
    cluster_ids = np.array([FIXED_CLUSTER_CODEBOOK[c] for c in clusters])
    qualities = [1] * n_events

    label = {
        "sr": sr,
        "spec_time_step": 0.01,
        "min_frequency": 0,
        "onset": onsets,
        "offset": offsets,
        "cluster": clusters,
        "cluster_id": cluster_ids,
        "quality": qualities,
    }
    return audio, label


# ---------------------------------------------------------------------------
# Tests: FIXED_CLUSTER_CODEBOOK / ID_TO_CLUSTER consistency
# ---------------------------------------------------------------------------

def test_codebook_and_id_mapping_are_consistent():
    """ID_TO_CLUSTER should map every value in FIXED_CLUSTER_CODEBOOK back."""
    for cluster_name, cluster_id in FIXED_CLUSTER_CODEBOOK.items():
        assert cluster_id in ID_TO_CLUSTER, (
            f"ID {cluster_id} (from '{cluster_name}') not in ID_TO_CLUSTER"
        )


def test_id_to_cluster_values_are_strings():
    for cid, name in ID_TO_CLUSTER.items():
        assert isinstance(name, str)
        assert isinstance(cid, int)


# ---------------------------------------------------------------------------
# Tests: slice_audios_and_labels
# ---------------------------------------------------------------------------

def test_slice_returns_correct_types():
    """Slicing should return lists of arrays, labels, and metadata dicts."""
    audio, label = _make_audio_and_label(duration_sec=5.0)
    audios, labels, metas = slice_audios_and_labels([audio], [label], 3000)

    assert isinstance(audios, list)
    assert isinstance(labels, list)
    assert isinstance(metas, list)
    assert len(audios) == len(labels) == len(metas)
    assert len(audios) > 0


def test_slice_metadata_has_required_keys():
    audio, label = _make_audio_and_label(duration_sec=5.0)
    _, _, metas = slice_audios_and_labels([audio], [label], 3000)

    for m in metas:
        assert "original_idx" in m
        assert "segment_idx" in m


def test_slice_no_event_loss():
    """The total number of events across slices should be >= the original count.

    Events at slice boundaries may appear in multiple slices (duplicated),
    but none should be lost entirely.
    """
    audio, label = _make_audio_and_label(duration_sec=5.0, n_events=4)
    n_original = len(label["onset"])

    _, labels_sliced, _ = slice_audios_and_labels([audio], [label], 3000)

    n_total = sum(len(l["onset"]) for l in labels_sliced)
    assert n_total >= n_original, (
        f"Expected at least {n_original} events across slices, got {n_total}"
    )


def test_slice_onsets_are_nonnegative():
    """After slicing, all onset times should be >= 0 (local to the slice)."""
    audio, label = _make_audio_and_label(duration_sec=5.0, n_events=5)
    _, labels_sliced, _ = slice_audios_and_labels([audio], [label], 3000)

    for l in labels_sliced:
        if len(l["onset"]) > 0:
            assert np.all(l["onset"] >= 0), f"Negative onset found: {l['onset']}"


def test_slice_offsets_greater_than_onsets():
    """Offset should always be > onset for every event."""
    audio, label = _make_audio_and_label(duration_sec=5.0, n_events=5)
    _, labels_sliced, _ = slice_audios_and_labels([audio], [label], 3000)

    for l in labels_sliced:
        if len(l["onset"]) > 0:
            assert np.all(l["offset"] > l["onset"]), (
                f"offset <= onset: {l['offset']} vs {l['onset']}"
            )


def test_slice_empty_audio():
    """Slicing an empty audio should return empty lists."""
    audio = np.array([], dtype=np.float32)
    label = {
        "sr": 16000, "spec_time_step": 0.01, "min_frequency": 0,
        "onset": np.array([]), "offset": np.array([]),
        "cluster": [], "cluster_id": np.array([]), "quality": [],
    }
    audios, labels, metas = slice_audios_and_labels([audio], [label], 3000)
    assert len(audios) == 0


def test_slice_short_audio_single_segment():
    """An audio shorter than one clip should produce exactly one segment."""
    audio, label = _make_audio_and_label(duration_sec=2.0, n_events=1)
    audios, labels, metas = slice_audios_and_labels([audio], [label], 3000)

    # 3000 cols * 0.01 s/col = 30 s clip; 2 s audio < 30 s -> 1 segment
    assert len(audios) == 1
