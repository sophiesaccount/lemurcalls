import json

import torch
from tqdm import tqdm

from ..util.common import is_scheduled_job
from ..util.confusion_framewise import confusion_matrix_framewise
from ..utils import *


def collate_fn(batch):
    """Stack batch items into tensors for input_features, decoder_input_ids, labels.

    Args:
        batch: List of sample dicts.

    Returns:
        dict: Batched tensors for input_features, decoder_input_ids, labels.
    """
    input_features = [item["input_features"].clone().detach().float() for item in batch]
    decoder_input_ids = [item["decoder_input_ids"].clone().detach().long() for item in batch]
    labels = [item["labels"].clone().detach().long() for item in batch]

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
    """Run one training step with mixed precision.

    Args:
        model: Model to train.
        batch: Batch dict (moved to device inside).
        optimizer: Optimizer.
        scheduler: LR scheduler (optional; not stepped here).
        scaler: GradScaler for AMP.
        device: Device to run on.

    Returns:
        float: Loss value for this step.
    """
    for key in batch:
        batch[key] = batch[key].to(device)
    
    optimizer.zero_grad()
    with torch.amp.autocast(device_type="cuda", dtype=torch.float16):   
        model_out = model(**batch)
        loss = model_out.loss.mean()
    
    scaler.scale(loss).backward()
    scaler.step(optimizer)
    # scheduler.step()  # Optional: call scheduler here if using per-iteration schedule
    scaler.update()
    
    return loss.item()

def evaluate(audio_list, label_list, segmenter, batch_size, max_length, num_trials, consolidation_method="clustering", num_beams=4, target_cluster=None, confusion_matrix: str = None, save_cm_data: str = None):
    """Run segmenter on each audio and aggregate segment-wise and frame-wise metrics.

    Args:
        audio_list: List of audio arrays.
        label_list: List of label dicts (with sr, min_frequency, spec_time_step, etc.).
        segmenter: Segmenter with .segment(), .segment_score(), .frame_score().
        batch_size: Batch size for segmentation.
        max_length: Max generation length.
        num_trials: Number of trials.
        consolidation_method: 'clustering' or voting method.
        num_beams: Beam size.
        target_cluster: If set, score only this cluster.
        confusion_matrix: Optional name for confusion matrix output.
        save_cm_data: Optional name to save raw prediction/label.

    Returns:
        dict: 'segment_wise' and 'frame_wise' lists [TP, P_pred, P_label, precision, recall, f1].
    """
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
        # Workaround to pass job-id via confusion_matrix and save_cm_data
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