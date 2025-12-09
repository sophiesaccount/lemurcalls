import argparse
import json
import os
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from whisperformer_dataset import WhisperFormerDataset
from whisperformer_model import WhisperFormer
#from wf_model_old import WhisperFormer
#from model_linear import WhisperFormer
#from transformers import WhisperModel
from datautils import get_audio_and_label_paths_from_folders, load_data, get_cluster_codebook
from datautils import slice_audios_and_labels
#from whisperformer_train import collate_fn  # Reuse collate function from training
import numpy as np
from collections import defaultdict


if __name__ == "__main__":
    print('hello')