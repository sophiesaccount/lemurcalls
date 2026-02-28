"""Tests for loss functions (focal loss, GIoU, DIoU)."""

import pytest
import torch

from lemurcalls.whisperformer.losses import (
    sigmoid_focal_loss,
    ctr_giou_loss_1d,
    ctr_diou_loss_1d,
)


# ---------------------------------------------------------------------------
# Focal Loss
# ---------------------------------------------------------------------------


class TestSigmoidFocalLoss:
    def test_perfect_prediction_low_loss(self):
        """Loss should be near zero when predictions perfectly match targets."""
        # Large positive logits for positive targets
        inputs = torch.tensor([10.0, 10.0, -10.0, -10.0])
        targets = torch.tensor([1.0, 1.0, 0.0, 0.0])
        loss = sigmoid_focal_loss(inputs, targets, reduction="mean")
        assert loss.item() < 0.01

    def test_wrong_prediction_high_loss(self):
        """Loss should be high when predictions are opposite to targets."""
        inputs = torch.tensor([-10.0, -10.0, 10.0, 10.0])
        targets = torch.tensor([1.0, 1.0, 0.0, 0.0])
        loss = sigmoid_focal_loss(inputs, targets, reduction="mean")
        assert loss.item() > 1.0

    def test_all_zeros_input(self):
        """Loss at logit=0 (sigmoid=0.5) should be moderate."""
        inputs = torch.zeros(4)
        targets = torch.ones(4)
        loss = sigmoid_focal_loss(inputs, targets, reduction="mean")
        assert loss.item() > 0

    def test_reduction_none(self):
        """With reduction='none', output shape should match input shape."""
        inputs = torch.randn(5, 3)
        targets = torch.randint(0, 2, (5, 3)).float()
        loss = sigmoid_focal_loss(inputs, targets, reduction="none")
        assert loss.shape == (5, 3)

    def test_reduction_sum_vs_none(self):
        inputs = torch.randn(4)
        targets = torch.randint(0, 2, (4,)).float()
        loss_none = sigmoid_focal_loss(inputs, targets, reduction="none")
        loss_sum = sigmoid_focal_loss(inputs, targets, reduction="sum")
        assert torch.isclose(loss_none.sum(), loss_sum, atol=1e-5)


# ---------------------------------------------------------------------------
# GIoU Loss
# ---------------------------------------------------------------------------


class TestGIoULoss:
    def test_identical_offsets_zero_loss(self):
        """Loss should be 0 when predicted and target offsets are identical."""
        offsets = torch.tensor([[1.0, 2.0], [3.0, 1.0]])
        loss = ctr_giou_loss_1d(offsets, offsets, reduction="mean")
        assert torch.isclose(loss, torch.tensor(0.0), atol=1e-5)

    def test_no_overlap_high_loss(self):
        """Loss should be ~1 when there is no overlap at all."""
        pred = torch.tensor([[0.0, 0.1]])  # tiny interval
        target = torch.tensor([[10.0, 10.0]])  # far away
        loss = ctr_giou_loss_1d(pred, target, reduction="mean")
        assert loss.item() > 0.5

    def test_nonnegative(self):
        """GIoU loss should be non-negative."""
        pred = torch.rand(10, 2)
        target = torch.rand(10, 2)
        loss = ctr_giou_loss_1d(pred, target, reduction="none")
        assert torch.all(loss >= -1e-6)


# ---------------------------------------------------------------------------
# DIoU Loss
# ---------------------------------------------------------------------------


class TestDIoULoss:
    def test_identical_offsets_zero_loss(self):
        offsets = torch.tensor([[2.0, 3.0], [1.0, 1.0]])
        loss = ctr_diou_loss_1d(offsets, offsets, reduction="mean")
        assert torch.isclose(loss, torch.tensor(0.0), atol=1e-5)

    def test_no_overlap_high_loss(self):
        pred = torch.tensor([[0.0, 0.1]])
        target = torch.tensor([[10.0, 10.0]])
        loss = ctr_diou_loss_1d(pred, target, reduction="mean")
        assert loss.item() > 0.5

    def test_symmetry(self):
        """DIoU(a, b) should equal DIoU(b, a) for same-center intervals."""
        a = torch.tensor([[1.0, 2.0]])
        b = torch.tensor([[2.0, 1.0]])
        loss_ab = ctr_diou_loss_1d(a, b, reduction="mean")
        loss_ba = ctr_diou_loss_1d(b, a, reduction="mean")
        assert torch.isclose(loss_ab, loss_ba, atol=1e-5)

    def test_empty_input(self):
        """Empty input should return zero loss."""
        pred = torch.zeros(0, 2)
        target = torch.zeros(0, 2)
        loss = ctr_diou_loss_1d(pred, target, reduction="sum")
        assert loss.item() == 0.0
