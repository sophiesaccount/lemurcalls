import torch
import torch.nn as nn

KERNEL_SIZE = 3


class WhisperEncoder(nn.Module):
    """Wraps a Whisper encoder for use in WhisperFormer."""

    def __init__(self, encoder):
        super().__init__()
        self.encoder = encoder

    def forward(self, x):
        """Forward pass through the encoder.

        Args:
            x: Input features (e.g. mel spectrogram).

        Returns:
            Encoder output (e.g. last_hidden_state).
        """
        return self.encoder(x)

class LightDecoderLayer(nn.Module):
    """Single transformer decoder layer with self-attention and feed-forward block."""

    def __init__(self, d_model, n_heads=4, dim_ff=None, dropout=0.1):
        super().__init__()

        if dim_ff is None:
            dim_ff = 2 * d_model

        assert d_model % n_heads == 0, \
            f"d_model={d_model} must be divisible by n_heads={n_heads}"

        self.self_attn = nn.MultiheadAttention(
            d_model, n_heads, batch_first=True
        )

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)

        self.dropout_attn = nn.Dropout(dropout)
        self.dropout_ff = nn.Dropout(dropout)

        self.ff = nn.Sequential(
            nn.Linear(d_model, dim_ff),
            nn.GELU(),
            nn.Linear(dim_ff, d_model)
        )

    def forward(self, x):
        """Forward pass: self-attention then feed-forward with residual connections."""
        attn_out, _ = self.self_attn(x, x, x)
        attn_out = self.dropout_attn(attn_out)
        x = x + attn_out
        x = self.norm1(x)

        ff_out = self.ff(x)
        ff_out = self.dropout_ff(ff_out)
        x = x + ff_out
        x = self.norm2(x)

        return x


class LightDecoder(nn.Module):
    """Stack of LightDecoderLayer with final LayerNorm."""

    def __init__(self, d_model, num_layers=3, n_heads=4, dim_ff=None, dropout=0.1):
        super().__init__()

        self.layers = nn.ModuleList([
            LightDecoderLayer(d_model, n_heads, dim_ff, dropout)
            for _ in range(num_layers)
        ])

        self.norm = nn.LayerNorm(d_model)

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return self.norm(x)


class ClassificationHead(nn.Module):
    """1D convolutional head for per-frame class logits (similar to ActionFormer)."""

    def __init__(self, input_dim, num_classes, num_layers=2, dropout=0.1):
        super().__init__()

        layers = []
        for i in range(num_layers):
            layers.append(nn.Conv1d(input_dim, input_dim, kernel_size=KERNEL_SIZE, padding=1))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout))

        self.layers = nn.Sequential(*layers)
        self.output_conv = nn.Conv1d(input_dim, num_classes, kernel_size=KERNEL_SIZE, padding=1)

    def forward(self, x):
        """Forward pass. Input (B, T, D) -> output (B, T, num_classes)."""
        x = x.transpose(1, 2)
        x = self.layers(x)
        x = self.output_conv(x)
        return x.transpose(1, 2)


class RegressionHead(nn.Module):
    """1D convolutional head for per-frame start/end offsets (similar to ActionFormer)."""

    def __init__(self, input_dim, num_layers=2, dropout=0.1):
        super().__init__()

        layers = []
        for i in range(num_layers):
            layers.append(nn.Conv1d(input_dim, input_dim, kernel_size=KERNEL_SIZE, padding=1))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout))

        self.layers = nn.Sequential(*layers)
        self.output_conv = nn.Conv1d(input_dim, 2, kernel_size=KERNEL_SIZE, padding=1)

    def forward(self, x):
        """Forward pass. Input (B, T, D) -> output (B, T, 2) non-negative offsets."""
        x = x.transpose(1, 2)
        x = self.layers(x)
        x = self.output_conv(x)
        x = torch.relu(x)
        return x.transpose(1, 2)


class WhisperFormer(nn.Module):
    """Whisper encoder + light decoder + classification and regression heads for event detection."""

    def __init__(
        self,
        encoder,
        num_classes,
        num_decoder_layers=3,
        num_head_layers=2,
        dropout=0.1
    ):
        super().__init__()

        self.encoder = WhisperEncoder(encoder)
        d_model = encoder.config.d_model

        self.decoder = LightDecoder(
            d_model=d_model,
            num_layers=num_decoder_layers,
            dropout=dropout
        )

        self.class_head = ClassificationHead(
            input_dim=d_model,
            num_classes=num_classes,
            num_layers=num_head_layers,
            dropout=dropout
        )

        self.regr_head = RegressionHead(
            input_dim=d_model,
            num_layers=num_head_layers,
            dropout=dropout
        )

    def forward(self, input_features):
        """Forward pass. Returns class logits and regression offsets per frame."""
        encoder_outputs = self.encoder(input_features)
        hidden_states = encoder_outputs.last_hidden_state

        decoder_outputs = self.decoder(hidden_states)

        class_preds = self.class_head(decoder_outputs)
        regr_preds = self.regr_head(decoder_outputs)

        return class_preds, regr_preds


def infer_architecture_from_state_dict(state_dict):
    """Infer num_decoder_layers, num_head_layers, and num_classes from checkpoint keys.

    Works with checkpoints that only store ``model.state_dict()`` (no metadata).

    Args:
        state_dict: The model's state dict (as returned by ``torch.load``).

    Returns:
        Tuple (num_decoder_layers, num_head_layers, num_classes).
    """
    decoder_indices = set()
    head_indices = set()
    num_classes = None

    for key in state_dict.keys():
        # Keys like "decoder.layers.0.self_attn.in_proj_weight"
        if key.startswith("decoder.layers."):
            idx = int(key.split(".")[2])
            decoder_indices.add(idx)
        # Keys like "class_head.layers.0.weight" (Conv1d, ReLU, Dropout per layer)
        if key.startswith("class_head.layers."):
            idx = int(key.split(".")[2])
            head_indices.add(idx)
        # "class_head.output_conv.weight" has shape (num_classes, d_model, kernel)
        if key == "class_head.output_conv.weight":
            num_classes = state_dict[key].shape[0]

    num_decoder_layers = len(decoder_indices)
    # ClassificationHead creates 3 nn.Modules per layer (Conv1d, ReLU, Dropout)
    num_head_layers = len(head_indices) // 3 if head_indices else 0

    return num_decoder_layers, num_head_layers, num_classes

