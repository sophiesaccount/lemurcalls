import torch
import torch.nn as nn
from transformers import WhisperModel

# 1. Load Whisper Encoder
whisper_model = WhisperModel.from_pretrained("openai/whisper-tiny")
INPUT_DIM = whisper_model.config.d_model  # e.g. 384 for whisper-tiny
NUM_CLASSES = 1


# 2. Classification Head mit Linear Layers
class ClassificationHead(nn.Module):
    def __init__(self, input_dim=INPUT_DIM, num_classes=NUM_CLASSES, hidden_dim=None):
        super().__init__()
        if hidden_dim is None:
            hidden_dim = input_dim  

        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, num_classes)
        )

    def forward(self, x):  # x: (B, T, D)
        return self.net(x)  # (B, T, C)


# 3. Regression Head mit Linear Layers
class RegressionHead(nn.Module):
    def __init__(self, input_dim=INPUT_DIM, hidden_dim=None):
        super().__init__()
        if hidden_dim is None:
            hidden_dim = input_dim

        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 2),
            nn.ReLU()   
        )

    def forward(self, x):  # x: (B, T, D)
        return self.net(x)  # (B, T, 2)


# 4. Gesamtes Modell
class WhisperFormer(nn.Module):
    def __init__(self, encoder, num_classes=NUM_CLASSES):
        super().__init__()
        self.encoder = encoder
        self.class_head = ClassificationHead(num_classes=num_classes)
        self.regr_head = RegressionHead()

    def forward(self, input_features):
        
        encoder_outputs = self.encoder(input_features)
        hidden_states = encoder_outputs.last_hidden_state  # (B, T, D)

        class_preds = self.class_head(hidden_states)  # (B, T, C)
        regr_preds = self.regr_head(hidden_states)    # (B, T, 2)

        return class_preds, regr_preds
