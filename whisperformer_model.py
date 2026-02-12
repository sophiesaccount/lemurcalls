import torch
import torch.nn as nn
from transformers import WhisperModel

KERNEL_SIZE = 3

# 1. Load Whisper Encoder
class WhisperEncoder(nn.Module):
    def __init__(self, encoder):
        super().__init__()
        self.encoder = encoder

    def forward(self, x):
        return self.encoder(x)

class LightDecoderLayer(nn.Module):
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
        # Self-Attention Block
        attn_out, _ = self.self_attn(x, x, x) 
        attn_out = self.dropout_attn(attn_out)
        x = x + attn_out                      # Skip-Connection
        x = self.norm1(x)                     # LayerNorm danach

        # Feed-Forward Block
        ff_out = self.ff(x)
        ff_out = self.dropout_ff(ff_out)
        x = x + ff_out                        # wieder Skip-Connection
        x = self.norm2(x)

        return x


class LightDecoder(nn.Module):
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




# 2. Classification Head (similar to ActionFormer) 
class ClassificationHead(nn.Module):
    def __init__(self, input_dim, num_classes, num_layers=2, dropout=0.1):
        super().__init__()

        layers = []
        for i in range(num_layers):
            layers.append(nn.Conv1d(input_dim, input_dim, kernel_size=KERNEL_SIZE, padding=1))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout))

        self.layers = nn.Sequential(*layers)
        self.output_conv = nn.Conv1d(input_dim, num_classes, kernel_size=KERNEL_SIZE, padding=1)

    def forward(self, x):  # (B, T, D)
        x = x.transpose(1, 2)      # → (B, D, T)
        x = self.layers(x)         # alle Conv-Blöcke
        x = self.output_conv(x)    # → (B, num_classes, T)
        return x.transpose(1, 2)   # → (B, T, C)


# 3. Regression Head (similar to ActionFormer)
class RegressionHead(nn.Module):
    def __init__(self, input_dim, num_layers=2, dropout=0.1):
        super().__init__()

        layers = []
        for i in range(num_layers):
            layers.append(nn.Conv1d(input_dim, input_dim, kernel_size=KERNEL_SIZE, padding=1))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout))

        self.layers = nn.Sequential(*layers)
        self.output_conv = nn.Conv1d(input_dim, 2, kernel_size=KERNEL_SIZE, padding=1)

    def forward(self, x):  # (B, T, D)
        x = x.transpose(1, 2)       # (B, D, T)
        x = self.layers(x)
        x = self.output_conv(x)
        x = torch.relu(x)
        return x.transpose(1, 2)    # (B, T, 2)


class WhisperFormer(nn.Module):
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
        d_model = encoder.config.d_model   # 🔥 EINZIGE Quelle der Wahrheit

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
        encoder_outputs = self.encoder(input_features)
        hidden_states = encoder_outputs.last_hidden_state  # (B, T/2, D)

        decoder_outputs = self.decoder(hidden_states)

        class_preds = self.class_head(decoder_outputs)
        regr_preds = self.regr_head(decoder_outputs)

        return class_preds, regr_preds

