import argparse
import json
import os
import sys
from datetime import datetime
from glob import glob

import numpy as np
import torch
import torch.nn as nn
import wandb
from torch.optim import AdamW
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import get_linear_schedule_with_warmup

from convert_hf_to_ct2 import convert_hf_to_ct2
from datautils import (VocalSegDataset, get_audio_and_label_paths,
                       get_cluster_codebook, load_data,
                       slice_audios_and_labels, train_val_split)
from model import WhisperSegmenterForEval, load_model, save_model
from util.common import EarlyStopHandler, is_scheduled_job
from util.confusion_framewise import confusion_matrix_framewise
from utils import *
from torch.nn.utils.rnn import pad_sequence


def collate_fn(batch):
    # batch is a list of samples (dicts)
    
    input_features = [torch.tensor(item["input_features"], dtype=torch.float32) for item in batch]
    decoder_input_ids = [torch.tensor(item["decoder_input_ids"], dtype=torch.int64) for item in batch]
    labels = [torch.tensor(item["labels"], dtype=torch.int64) for item in batch]

    # Stack tensors along the first dimension (batch dimension)
    input_features = torch.stack(input_features)
    decoder_input_ids = torch.stack(decoder_input_ids)
    labels = torch.stack(labels)

    return {
        "input_features": input_features,
        "decoder_input_ids": decoder_input_ids,
        "labels": labels,
    }

def train_iteration(model, batch, optimizer, scheduler, scaler, device):
    for key in batch:
        batch[key] = batch[key].to(device)
    
    optimizer.zero_grad()
    with torch.amp.autocast(device_type="cuda", dtype=torch.float16):   
        model_out = model(**batch)
        loss = model_out.loss.mean()
    
    scaler.scale(loss).backward()
    scaler.step(optimizer)
    #scheduler.step()  # optional: call scheduler here if you're using it per iteration
    scaler.update()
    
    return loss.item()

def evaluate( audio_list, label_list, segmenter, batch_size, max_length, num_trials, consolidation_method = "clustering", num_beams=4, target_cluster = None, confusion_matrix: str = None, save_cm_data: str = None):

    total_n_true_positive_segment_wise, total_n_positive_in_prediction_segment_wise, total_n_positive_in_label_segment_wise = 0,0,0
    total_n_true_positive_frame_wise, total_n_positive_in_prediction_frame_wise, total_n_positive_in_label_frame_wise = 0,0,0

    for audio, label in tqdm(zip(audio_list, label_list), total = len(audio_list), desc = "evaluate()", disable=is_scheduled_job()):
        prediction = segmenter.segment(
            audio, sr = label["sr"],
            min_frequency = label["min_frequency"],
            spec_time_step = label["spec_time_step"],
            min_segment_length = label["min_segment_length"],
            eps = label["eps"],  ## for DBSCAN clustering
            time_per_frame_for_voting = label.get("time_per_frame_for_voting", 0.001), ## for bin-wise voting, by default it is not used
            consolidation_method = consolidation_method,
            max_length = max_length, 
            batch_size = batch_size, 
            num_trials = num_trials,
            num_beams = num_beams
        )
        # dirty workaround to pass the job-id in `confusion_matrix` and `save_cm_data
        if confusion_matrix != None:
            confusion_matrix_framewise(prediction, label, None, label["time_per_frame_for_scoring"], name=confusion_matrix)
        if save_cm_data != None:
            with open(f'/usr/users/bhenne/projects/whisperseg/results/{save_cm_data}.cmraw', "w") as fp:
                json.dump({'prediction': prediction, 'label': label}, fp, indent=2)
        TP, P_pred, P_label = segmenter.segment_score( prediction, label,  target_cluster = target_cluster, tolerance = label["tolerance"] )[:3]
        total_n_true_positive_segment_wise += TP
        total_n_positive_in_prediction_segment_wise += P_pred
        total_n_positive_in_label_segment_wise += P_label
        
        
        TP, P_pred, P_label = segmenter.frame_score( prediction, label,  target_cluster = target_cluster, 
                                                     time_per_frame_for_scoring = label["time_per_frame_for_scoring"] )[:3]
        
        total_n_true_positive_frame_wise += TP
        total_n_positive_in_prediction_frame_wise += P_pred
        total_n_positive_in_label_frame_wise += P_label
        
    res = {}
    
    precision = total_n_true_positive_segment_wise / max(total_n_positive_in_prediction_segment_wise, 1e-12)
    recall = total_n_true_positive_segment_wise / max( total_n_positive_in_label_segment_wise, 1e-12 )
    f1 = 2/(1/max(precision, 1e-12) + 1/max(recall, 1e-12)  )
    
    res["segment_wise"] = [ total_n_true_positive_segment_wise, total_n_positive_in_prediction_segment_wise, total_n_positive_in_label_segment_wise, precision, recall, f1 ]
    
    
    precision = total_n_true_positive_frame_wise / max(total_n_positive_in_prediction_frame_wise, 1e-12)
    recall = total_n_true_positive_frame_wise / max( total_n_positive_in_label_frame_wise, 1e-12 )
    f1 = 2/(1/max(precision, 1e-12) + 1/max(recall, 1e-12)  )
    
    res["frame_wise"] = [ total_n_true_positive_frame_wise, total_n_positive_in_prediction_frame_wise, total_n_positive_in_label_frame_wise, precision, recall, f1 ]
    
    return res