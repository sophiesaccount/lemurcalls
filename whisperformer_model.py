import torch
import torch.nn as nn
from transformers import WhisperModel


NUM_CLASSES = 10  # number of classes TODO: make this an input 
INPUT_DIM = 384   # Whisper encoder hidden size (for small model)
KERNEL_SIZE = 3


# 1. Load Whisper Encoder
whisper_model = WhisperModel.from_pretrained("openai/whisper-tiny")
encoder = whisper_model.encoder


# 2. Classification Head (similar to ActionFormer) 
class ClassificationHead(nn.Module):
    def __init__(self, input_dim=INPUT_DIM, num_classes=NUM_CLASSES):
        super().__init__()
        self.conv1 = nn.Conv1d(input_dim, input_dim, kernel_size=KERNEL_SIZE, padding=1)
        self.norm1 = nn.LayerNorm(input_dim)
        self.conv2 = nn.Conv1d(input_dim, input_dim, kernel_size=KERNEL_SIZE, padding=1)
        self.norm2 = nn.LayerNorm(input_dim)
        self.conv3 = nn.Conv1d(input_dim, num_classes, kernel_size=KERNEL_SIZE, padding=1)

    def forward(self, x):  # x: (B, T, D)=(batch_size, sequence_length, hidden_size)=(4,3000,384?)
        x = x.transpose(1, 2)  # (B, D, T)
        x = self.conv1(x).transpose(1, 2)  # (B, T, D)
        x = self.norm1(x).transpose(1, 2)  # (B, D, T)
        x = torch.relu(x)

        x = self.conv2(x).transpose(1, 2)
        x = self.norm2(x).transpose(1, 2)
        x = torch.relu(x)

        x = self.conv3(x)  # (B, num_classes, T) = (B,T,C)
        x = torch.sigmoid(x) #values between 0 and 1
        return x.transpose(1, 2)  # → (B, T, C)


# 3. Regression Head (similar to ActionFormer)
class RegressionHead(nn.Module):
    def __init__(self, input_dim=INPUT_DIM):
        super().__init__()
        self.conv1 = nn.Conv1d(input_dim, input_dim, kernel_size=KERNEL_SIZE, padding=1)
        self.norm1 = nn.LayerNorm(input_dim)
        self.conv2 = nn.Conv1d(input_dim, input_dim, kernel_size=KERNEL_SIZE, padding=1)
        self.norm2 = nn.LayerNorm(input_dim)
        self.conv3 = nn.Conv1d(input_dim, 2, kernel_size=KERNEL_SIZE, padding=1)  # Output: (B, 2, T)

    def forward(self, x):  # x: (B, T, D)
        x = x.transpose(1, 2)               # → (B, D, T)
        x = self.conv1(x).transpose(1, 2)   # → (B, T, D)
        x = self.norm1(x).transpose(1, 2)   # → (B, D, T)
        x = torch.relu(x)

        x = self.conv2(x).transpose(1, 2)
        x = self.norm2(x).transpose(1, 2)
        x = torch.relu(x)

        x = self.conv3(x)                   # → (B, 2, T) 
        x = torch.relu(x)                   # ReLU for positive distances
        return x.transpose(1, 2)            # → (B, T, 2)


# 4.entier model
class WhisperFormer(nn.Module):
    def __init__(self, encoder, num_classes=NUM_CLASSES):
        super().__init__()
        self.encoder = encoder
        self.class_head = ClassificationHead(num_classes=num_classes)
        self.regr_head = RegressionHead()

    def forward(self, input_features):
        """
        input_features: dict from Whisper feature extractor or embedded tokens
        Should have shape (B, T, D)
        """
        encoder_outputs = self.encoder(input_features) #not a tensor, but a model output
        hidden_states = encoder_outputs.last_hidden_state  # (B, T, D)

        class_preds = self.class_head(hidden_states)  # (B, T, C)
        regr_preds = self.regr_head(hidden_states)    # (B, T, 2)

        return class_preds, regr_preds

