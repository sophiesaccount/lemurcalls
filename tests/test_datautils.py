"""Tests for the data loading and slicing pipeline."""

import numpy as np
import pytest

from lemurcalls.datautils import (
    FIXED_CLUSTER_CODEBOOK,
    ID_TO_CLUSTER,
    slice_audios_and_labels,
)
from lemurcalls.whisperformer.datautils import get_codebook_for_classes
from lemurcalls.whisperseg.datautils_ben import (
    get_codebook_for_classes as get_codebook_for_classes_seg,
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
        "sr": 16000,
        "spec_time_step": 0.01,
        "min_frequency": 0,
        "onset": np.array([]),
        "offset": np.array([]),
        "cluster": [],
        "cluster_id": np.array([]),
        "quality": [],
    }
    audios, labels, metas = slice_audios_and_labels([audio], [label], 3000)
    assert len(audios) == 0


def test_slice_short_audio_single_segment():
    """An audio shorter than one clip should produce exactly one segment."""
    audio, label = _make_audio_and_label(duration_sec=2.0, n_events=1)
    audios, labels, metas = slice_audios_and_labels([audio], [label], 3000)

    # 3000 cols * 0.01 s/col = 30 s clip; 2 s audio < 30 s -> 1 segment
    assert len(audios) == 1


# ---------------------------------------------------------------------------
# Tests: get_codebook_for_classes
# ---------------------------------------------------------------------------


class TestGetCodebookForClasses:
    """Tests for the num_classes-dependent codebook factory."""

    def test_three_classes_returns_three_distinct_ids(self):
        codebook, id_map = get_codebook_for_classes(3)
        assert len(set(codebook.values())) == 3
        assert set(id_map.keys()) == {0, 1, 2}

    def test_three_classes_codebook_matches_fixed(self):
        """classes=3 should reproduce the original FIXED_CLUSTER_CODEBOOK."""
        codebook, _ = get_codebook_for_classes(3)
        assert codebook == {"m": 0, "t": 1, "w": 2, "lt": 1, "h": 1}

    def test_three_classes_id_map_covers_all_codebook_values(self):
        codebook, id_map = get_codebook_for_classes(3)
        for cluster_name, cid in codebook.items():
            assert cid in id_map, (
                f"class id {cid} (from '{cluster_name}') missing in id_to_cluster"
            )

    def test_one_class_maps_everything_to_zero(self):
        codebook, id_map = get_codebook_for_classes(1)
        assert all(v == 0 for v in codebook.values())
        assert set(id_map.keys()) == {0}

    def test_one_class_has_same_keys_as_three(self):
        """Single-class codebook should still know all cluster names."""
        cb1, _ = get_codebook_for_classes(1)
        cb3, _ = get_codebook_for_classes(3)
        assert set(cb1.keys()) == set(cb3.keys())

    def test_one_class_id_map_returns_moan(self):
        _, id_map = get_codebook_for_classes(1)
        assert id_map[0] == "m"

    def test_unsupported_value_raises_error(self):
        with pytest.raises(ValueError, match="not supported"):
            get_codebook_for_classes(2)

    def test_unsupported_zero_raises_error(self):
        with pytest.raises(ValueError, match="not supported"):
            get_codebook_for_classes(0)

    def test_unsupported_negative_raises_error(self):
        with pytest.raises(ValueError, match="not supported"):
            get_codebook_for_classes(-1)

    def test_return_types(self):
        codebook, id_map = get_codebook_for_classes(3)
        assert isinstance(codebook, dict)
        assert isinstance(id_map, dict)
        for k, v in codebook.items():
            assert isinstance(k, str)
            assert isinstance(v, int)
        for k, v in id_map.items():
            assert isinstance(k, int)
            assert isinstance(v, str)


# ---------------------------------------------------------------------------
# Tests: get_codebook_for_classes (WhisperSeg variant)
# ---------------------------------------------------------------------------


class TestGetCodebookForClassesWhisperSeg:
    """Tests for the WhisperSeg num_classes-dependent codebook factory."""

    def test_three_classes_returns_three_distinct_ids(self):
        codebook = get_codebook_for_classes_seg(3)
        assert len(set(codebook.values())) == 3

    def test_three_classes_codebook_values(self):
        codebook = get_codebook_for_classes_seg(3)
        assert codebook == {"m": 0, "t": 1, "w": 2, "lt": 1, "h": 1}

    def test_one_class_maps_everything_to_zero(self):
        codebook = get_codebook_for_classes_seg(1)
        assert all(v == 0 for v in codebook.values())

    def test_one_class_has_same_keys_as_three(self):
        """Single-class codebook should know all the same cluster names."""
        cb1 = get_codebook_for_classes_seg(1)
        cb3 = get_codebook_for_classes_seg(3)
        assert set(cb1.keys()) == set(cb3.keys())

    def test_unsupported_value_raises_error(self):
        with pytest.raises(ValueError, match="not supported"):
            get_codebook_for_classes_seg(2)

    def test_unsupported_zero_raises_error(self):
        with pytest.raises(ValueError, match="not supported"):
            get_codebook_for_classes_seg(0)

    def test_codebooks_consistent_between_whisperformer_and_whisperseg(self):
        """Both variants should produce the same cluster_codebook for classes=3."""
        cb_former, _ = get_codebook_for_classes(3)
        cb_seg = get_codebook_for_classes_seg(3)
        assert cb_former == cb_seg

    def test_return_type_is_dict(self):
        codebook = get_codebook_for_classes_seg(3)
        assert isinstance(codebook, dict)
        for k, v in codebook.items():
            assert isinstance(k, str)
            assert isinstance(v, int)
