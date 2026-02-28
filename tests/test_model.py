"""Tests for WhisperFormer model forward pass and shape consistency."""

import pytest
import torch
from transformers import WhisperModel, WhisperConfig


from lemurcalls.whisperformer.model import (
    WhisperFormer,
    ClassificationHead,
    RegressionHead,
    LightDecoder,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def whisper_base_encoder():
    """Create a Whisper Base encoder from config only (no pretrained weights)."""
    config = WhisperConfig(
        d_model=512,
        encoder_layers=2,  # small for fast tests
        encoder_attention_heads=8,
        encoder_ffn_dim=2048,
        decoder_layers=2,
        decoder_attention_heads=8,
        decoder_ffn_dim=2048,
    )
    model = WhisperModel(config)
    return model.encoder


# ---------------------------------------------------------------------------
# Tests: Individual components
# ---------------------------------------------------------------------------


class TestClassificationHead:
    def test_output_shape(self):
        B, T, D, C = 2, 1500, 512, 3
        head = ClassificationHead(input_dim=D, num_classes=C, num_layers=2)
        x = torch.randn(B, T, D)
        out = head(x)
        assert out.shape == (B, T, C)


class TestRegressionHead:
    def test_output_shape(self):
        B, T, D = 2, 1500, 512
        head = RegressionHead(input_dim=D, num_layers=2)
        x = torch.randn(B, T, D)
        out = head(x)
        assert out.shape == (B, T, 2)

    def test_output_nonnegative(self):
        """Regression offsets should be non-negative (ReLU output)."""
        B, T, D = 2, 1500, 512
        head = RegressionHead(input_dim=D, num_layers=2)
        x = torch.randn(B, T, D)
        out = head(x)
        assert torch.all(out >= 0)


class TestLightDecoder:
    def test_output_shape_preserved(self):
        B, T, D = 2, 1500, 512
        decoder = LightDecoder(d_model=D, num_layers=2)
        x = torch.randn(B, T, D)
        out = decoder(x)
        assert out.shape == (B, T, D)


# ---------------------------------------------------------------------------
# Tests: Full WhisperFormer
# ---------------------------------------------------------------------------


class TestWhisperFormer:
    def test_forward_shapes_base(self, whisper_base_encoder):
        """WhisperFormer with Base encoder should produce correct output shapes."""
        num_classes = 3
        model = WhisperFormer(
            encoder=whisper_base_encoder,
            num_classes=num_classes,
            num_decoder_layers=1,
            num_head_layers=1,
        )
        model.eval()

        B = 2
        # Whisper expects 80 mel bins, 3000 frames
        x = torch.randn(B, 80, 3000)

        with torch.no_grad():
            cls_out, reg_out = model(x)

        # Whisper encoder downsamples 3000 -> 1500
        T = 1500
        assert cls_out.shape == (B, T, num_classes), f"Got {cls_out.shape}"
        assert reg_out.shape == (B, T, 2), f"Got {reg_out.shape}"

    def test_regression_nonnegative(self, whisper_base_encoder):
        """Regression output should be non-negative."""
        model = WhisperFormer(
            encoder=whisper_base_encoder,
            num_classes=3,
            num_decoder_layers=1,
            num_head_layers=1,
        )
        model.eval()

        x = torch.randn(1, 80, 3000)
        with torch.no_grad():
            _, reg_out = model(x)

        assert torch.all(reg_out >= 0)

    def test_different_num_classes(self, whisper_base_encoder):
        """Model should work with different numbers of output classes."""
        for num_classes in [1, 3, 8]:
            model = WhisperFormer(
                encoder=whisper_base_encoder,
                num_classes=num_classes,
                num_decoder_layers=1,
                num_head_layers=1,
            )
            model.eval()
            x = torch.randn(1, 80, 3000)
            with torch.no_grad():
                cls_out, _ = model(x)
            assert cls_out.shape[-1] == num_classes
