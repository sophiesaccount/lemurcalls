import torch
import torch.nn as nn
from transformers import WhisperModel

whisper_model = WhisperModel.from_pretrained("openai/whisper-small", local_files_only=True)
NUM_CLASSES = 1  # number of classes TODO: make this an input 
INPUT_DIM = whisper_model.config.d_model  # Whisper encoder hidden size (for small model) 
KERNEL_SIZE = 3

# 1. Load Whisper Encoder

encoder = whisper_model.encoder

# To-Do: light weight decoder

# Lightweight decoder
class LightDecoderLayer(nn.Module):
    def __init__(self, d_model=INPUT_DIM, n_heads=4, dim_ff= 2*INPUT_DIM, dropout=0.1):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(d_model, n_heads, batch_first=True)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout_attn = nn.Dropout(dropout)  # Dropout nach Attention
        self.dropout_ff = nn.Dropout(dropout)    # Dropout nach Feed-Forward
        self.ff = nn.Sequential(
            nn.Linear(d_model, dim_ff),
            nn.GELU(),
            nn.Linear(dim_ff, d_model)
        )

    def forward(self, x):
        # Self-Attention Block
        attn_out, _ = self.self_attn(x, x, x)
        attn_out = self.dropout_attn(attn_out)
        x = x + attn_out
        x = self.norm1(x)

        # Feed-Forward Block
        ff_out = self.ff(x)
        ff_out = self.dropout_ff(ff_out)
        x = x + ff_out
        x = self.norm2(x)

        return x


class LightDecoder(nn.Module):
    def __init__(self, num_layers=3, d_model= INPUT_DIM, n_heads=4, dim_ff=2*INPUT_DIM, dropout=0.1):
        super().__init__()
        self.layers = nn.ModuleList([
            LightDecoderLayer(d_model, n_heads, dim_ff,dropout)
            for _ in range(num_layers)
        ])
        self.norm = nn.LayerNorm(d_model)  

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return self.norm(x)


# 2. Classification Head (similar to ActionFormer) 
class ClassificationHead(nn.Module):
    def __init__(self, input_dim=INPUT_DIM, num_classes=NUM_CLASSES, dropout=0.1):
        super().__init__()
        self.conv1 = nn.Conv1d(input_dim, input_dim, kernel_size=KERNEL_SIZE, padding=1)
        self.norm1 = nn.LayerNorm(input_dim)
        self.dropout1 = nn.Dropout(dropout)

        self.conv2 = nn.Conv1d(input_dim, input_dim, kernel_size=KERNEL_SIZE, padding=1)
        self.norm2 = nn.LayerNorm(input_dim)
        self.dropout2 = nn.Dropout(dropout)

        self.conv3 = nn.Conv1d(input_dim, num_classes, kernel_size=KERNEL_SIZE, padding=1)

    def forward(self, x):  # x: (B, T, D)=(batch_size, sequence_length, hidden_size)=(4,3000,384?)
        x = x.transpose(1, 2)  # (B, D, T)
        x = self.conv1(x).transpose(1, 2)  # (B, T, D)
        x = self.norm1(x).transpose(1, 2)  # (B, D, T)
        x = torch.relu(x)
        x = self.dropout1(x)

        x = self.conv2(x).transpose(1, 2)
        x = self.norm2(x).transpose(1, 2)
        x = torch.relu(x)
        x = self.dropout2(x)


        x = self.conv3(x)  # (B, num_classes, T) = (B,T,C)
        # #values between 0 and 1
        return x.transpose(1, 2)  # → (B, T, C)


# 3. Regression Head (similar to ActionFormer)
class RegressionHead(nn.Module):
    def __init__(self, input_dim=INPUT_DIM, dropout=0.1):
        super().__init__()
        self.conv1 = nn.Conv1d(input_dim, input_dim, kernel_size=KERNEL_SIZE, padding=1)
        self.norm1 = nn.LayerNorm(input_dim)
        self.dropout1 = nn.Dropout(dropout)

        self.conv2 = nn.Conv1d(input_dim, input_dim, kernel_size=KERNEL_SIZE, padding=1)
        self.norm2 = nn.LayerNorm(input_dim)
        self.dropout2 = nn.Dropout(dropout)

        self.conv3 = nn.Conv1d(input_dim, 2, kernel_size=KERNEL_SIZE, padding=1)  # Output: (B, 2, T)

    def forward(self, x):  # x: (B, T, D)
        x = x.transpose(1, 2)               # → (B, D, T)
        x = self.conv1(x).transpose(1, 2)   # → (B, T, D)
        x = self.norm1(x).transpose(1, 2)   # → (B, D, T)
        x = torch.relu(x)
        x = self.dropout1(x)

        x = self.conv2(x).transpose(1, 2)
        x = self.norm2(x).transpose(1, 2)
        x = torch.relu(x)
        x = self.dropout2(x)

        x = self.conv3(x)                  # → (B, 2, T)
        x = torch.relu(x)

        return x.transpose(1, 2)           # → (B, T, 2)


# 4.entier model
class WhisperFormer(nn.Module):
    def __init__(self, encoder, num_classes=NUM_CLASSES, no_decoder=False):
        super().__init__()
        self.encoder = encoder
        self.no_decoder = no_decoder
        print(no_decoder)

        self.decoder = LightDecoder()

        
        """
        if not no_decoder:
            self.decoder = LightDecoder()
            print('Using Decoder!')
        else:
            self.decoder = nn.Identity()
            print('Not using Decoder!')
        """

        self.class_head = ClassificationHead(num_classes=num_classes)
        self.regr_head = RegressionHead()

    def forward(self, input_features):
        """
        input_features: dict from Whisper feature extractor or embedded tokens
        Should have shape (B, T, D)
        """
        encoder_outputs = self.encoder(input_features) #not a tensor, but a model output
        hidden_states = encoder_outputs.last_hidden_state  # (B, T/2, D)

        decoder_outputs = self.decoder(hidden_states)  #(B, T/2, D)

        class_preds = self.class_head(decoder_outputs)  # (B, T/2, C)
        regr_preds = self.regr_head(decoder_outputs)    # (B, T/2, 2)

        return class_preds, regr_preds

