"""Tests for inference post-processing (NMS, prediction reconstruction)."""

import pytest
import torch

from lemurcalls.whisperformer.train import nms_1d_torch


# ---------------------------------------------------------------------------
# Tests: nms_1d_torch
# ---------------------------------------------------------------------------

class TestNMS1D:

    def test_empty_input(self):
        """NMS on empty tensor should return empty tensor with shape (0, 3)."""
        intervals = torch.zeros(0, 3)
        result = nms_1d_torch(intervals, iou_threshold=0.5)
        assert result.shape == (0, 3)

    def test_single_interval(self):
        """A single interval should be kept as-is."""
        intervals = torch.tensor([[1.0, 3.0, 0.9]])
        result = nms_1d_torch(intervals, iou_threshold=0.5)
        assert result.shape == (1, 3)
        assert torch.allclose(result, intervals)

    def test_non_overlapping_intervals(self):
        """Non-overlapping intervals should all be kept."""
        intervals = torch.tensor([
            [0.0, 1.0, 0.9],
            [2.0, 3.0, 0.8],
            [4.0, 5.0, 0.7],
        ])
        result = nms_1d_torch(intervals, iou_threshold=0.5)
        assert result.shape[0] == 3

    def test_identical_intervals_suppressed(self):
        """Identical intervals (IoU=1) should be reduced to one."""
        intervals = torch.tensor([
            [1.0, 3.0, 0.9],
            [1.0, 3.0, 0.8],
            [1.0, 3.0, 0.7],
        ])
        result = nms_1d_torch(intervals, iou_threshold=0.5)
        assert result.shape[0] == 1
        # The highest-scored interval should be kept
        assert torch.isclose(result[0, 2], torch.tensor(0.9))

    def test_high_iou_threshold_keeps_more(self):
        """A very high IoU threshold should suppress fewer intervals."""
        # Two overlapping intervals
        intervals = torch.tensor([
            [0.0, 2.0, 0.9],
            [1.0, 3.0, 0.8],
        ])
        # With high threshold (0.99), both should be kept (IoU < 0.99)
        result_high = nms_1d_torch(intervals, iou_threshold=0.99)
        # With low threshold (0.1), one should be suppressed
        result_low = nms_1d_torch(intervals, iou_threshold=0.1)

        assert result_high.shape[0] >= result_low.shape[0]

    def test_output_sorted_by_score_descending(self):
        """The kept intervals should be ordered by score (descending)."""
        intervals = torch.tensor([
            [0.0, 1.0, 0.3],
            [5.0, 6.0, 0.9],
            [10.0, 11.0, 0.6],
        ])
        result = nms_1d_torch(intervals, iou_threshold=0.5)
        scores = result[:, 2]
        for i in range(len(scores) - 1):
            assert scores[i] >= scores[i + 1]

    def test_output_shape_always_2d(self):
        """Output should always be 2D [N, 3], even for a single result."""
        intervals = torch.tensor([
            [0.0, 2.0, 0.9],
            [0.5, 2.5, 0.3],
        ])
        result = nms_1d_torch(intervals, iou_threshold=0.3)
        assert result.ndim == 2
        assert result.shape[1] == 3
