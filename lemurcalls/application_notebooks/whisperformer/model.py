import torch
import torch.nn as nn

KERNEL_SIZE = 3


class WhisperEncoder(nn.Module):
    """Wraps a Whisper encoder for use in WhisperFormer."""

    def __init__(self, encoder):
        super().__init__()
        self.encoder = encoder

    def forward(self, x):
        return self.encoder(x)


class LightDecoderLayer(nn.Module):
    """Single transformer decoder layer with self-attention and feed-forward block."""

    def __init__(self, d_model, n_heads=4, dim_ff=None, dropout=0.1):
        super().__init__()
        if dim_ff is None:
            dim_ff = 2 * d_model
        self.self_attn = nn.MultiheadAttention(d_model, n_heads, batch_first=True)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout_attn = nn.Dropout(dropout)
        self.dropout_ff = nn.Dropout(dropout)
        self.ff = nn.Sequential(
            nn.Linear(d_model, dim_ff),
            nn.GELU(),
            nn.Linear(dim_ff, d_model),
        )

    def forward(self, x):
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
        self.layers = nn.ModuleList(
            [
                LightDecoderLayer(d_model, n_heads, dim_ff, dropout)
                for _ in range(num_layers)
            ]
        )
        self.norm = nn.LayerNorm(d_model)

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return self.norm(x)


class ClassificationHead(nn.Module):
    """1D convolutional head for per-frame class logits."""

    def __init__(self, input_dim, num_classes, num_layers=2, dropout=0.1):
        super().__init__()
        layers = []
        for _ in range(num_layers):
            layers.append(
                nn.Conv1d(input_dim, input_dim, kernel_size=KERNEL_SIZE, padding=1)
            )
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout))
        self.layers = nn.Sequential(*layers)
        self.output_conv = nn.Conv1d(
            input_dim, num_classes, kernel_size=KERNEL_SIZE, padding=1
        )

    def forward(self, x):
        x = x.transpose(1, 2)
        x = self.layers(x)
        x = self.output_conv(x)
        return x.transpose(1, 2)


class RegressionHead(nn.Module):
    """1D convolutional head for per-frame start/end offsets."""

    def __init__(self, input_dim, num_layers=2, dropout=0.1):
        super().__init__()
        layers = []
        for _ in range(num_layers):
            layers.append(
                nn.Conv1d(input_dim, input_dim, kernel_size=KERNEL_SIZE, padding=1)
            )
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout))
        self.layers = nn.Sequential(*layers)
        self.output_conv = nn.Conv1d(input_dim, 2, kernel_size=KERNEL_SIZE, padding=1)

    def forward(self, x):
        x = x.transpose(1, 2)
        x = self.layers(x)
        x = self.output_conv(x)
        x = torch.relu(x)
        return x.transpose(1, 2)


class WhisperFormer(nn.Module):
    """Whisper encoder + light decoder + classification/regression heads."""

    def __init__(
        self, encoder, num_classes, num_decoder_layers=3, num_head_layers=2, dropout=0.1
    ):
        super().__init__()
        self.encoder = WhisperEncoder(encoder)
        d_model = encoder.config.d_model
        self.decoder = LightDecoder(
            d_model=d_model, num_layers=num_decoder_layers, dropout=dropout
        )
        self.class_head = ClassificationHead(
            input_dim=d_model,
            num_classes=num_classes,
            num_layers=num_head_layers,
            dropout=dropout,
        )
        self.regr_head = RegressionHead(
            input_dim=d_model, num_layers=num_head_layers, dropout=dropout
        )

    def forward(self, input_features):
        encoder_outputs = self.encoder(input_features)
        hidden_states = encoder_outputs.last_hidden_state
        decoder_outputs = self.decoder(hidden_states)
        class_preds = self.class_head(decoder_outputs)
        regr_preds = self.regr_head(decoder_outputs)
        return class_preds, regr_preds


def infer_architecture_from_state_dict(state_dict):
    """Infer num_decoder_layers, num_head_layers, and num_classes from checkpoint keys."""
    decoder_indices = set()
    head_conv_indices = set()
    num_classes = None
    for key in state_dict.keys():
        if key.startswith("decoder.layers."):
            decoder_indices.add(int(key.split(".")[2]))
        if key.startswith("class_head.layers.") and key.endswith(".weight"):
            head_conv_indices.add(int(key.split(".")[2]))
        if key == "class_head.output_conv.weight":
            num_classes = state_dict[key].shape[0]
    return len(decoder_indices), len(head_conv_indices), num_classes


def detect_whisper_size_from_state_dict(state_dict):
    """Detect Whisper model size ("base" or "large") from checkpoint weights."""
    for key in state_dict.keys():
        if "conv1.weight" in key:
            d_model = state_dict[key].shape[0]
            if d_model == 1280:
                return "large"
            elif d_model == 512:
                return "base"
    return None
