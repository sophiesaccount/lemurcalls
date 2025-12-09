import argparse
import json
import os
import sys
from datetime import datetime
from glob import glob
import matplotlib.pyplot as plt
import csv
import numpy as np
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import get_linear_schedule_with_warmup

from convert_hf_to_ct2 import convert_hf_to_ct2
from datautils import (VocalSegDataset, get_audio_and_label_paths, get_audio_and_label_paths_from_folders,
                       get_cluster_codebook, load_data,
                       slice_audios_and_labels, train_val_split, FIXED_CLUSTER_CODEBOOK, ID_TO_CLUSTER)

from util.common import EarlyStopHandler, is_scheduled_job
from util.confusion_framewise import confusion_matrix_framewise
from utils import *
from torch.nn.utils.rnn import pad_sequence
from whisperformer_dataset_quality import WhisperFormerDatasetQuality
from whisperformer_model import WhisperFormer
from losses import sigmoid_focal_loss, ctr_diou_loss_1d
from torch.cuda.amp import autocast, GradScaler
from torch.nn import functional as F
from torch.optim.lr_scheduler import ReduceLROnPlateau
from transformers import WhisperFeatureExtractor
from sklearn.model_selection import train_test_split
import copy
from collections import defaultdict
import random
import contextlib
from collections import defaultdict


SEED = 66100
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

def soft_nms_1d_torch(intervals: torch.Tensor, iou_threshold=0.5, sigma=0.5, score_threshold=0.001, method='gaussian'):
    """
    Soft-NMS for 1D intervals.
    
    intervals: Tensor [N, 3] -> (start, end, score)
    iou_threshold: IoU threshold for linear method
    sigma: sigma for gaussian method
    score_threshold: remove intervals with score < threshold
    method: 'linear' or 'gaussian'
    
    returns: Tensor [M, 3] of kept intervals
    """
    if intervals.numel() == 0:
        return intervals.new_zeros((0,3))

    intervals = intervals.clone()
    keep = []

    while intervals.numel() > 0:
        # get the interval with the max score
        max_idx = torch.argmax(intervals[:,2])
        current = intervals[max_idx]
        keep.append(current)

        if intervals.size(0) == 1:
            break

        rest = torch.cat([intervals[:max_idx], intervals[max_idx+1:]], dim=0)

        # IoU calculation
        ss = torch.maximum(current[0], rest[:,0])
        ee = torch.minimum(current[1], rest[:,1])
        inter = torch.clamp(ee - ss, min=0)
        union = (current[1]-current[0]) + (rest[:,1]-rest[:,0]) - inter
        iou = inter / union

        # update scores
        if method == 'linear':
            decay = torch.ones_like(iou)
            mask = iou > iou_threshold
            decay[mask] = 1 - iou[mask]
            rest[:,2] *= decay
        elif method == 'gaussian':
            rest[:,2] *= torch.exp(- (iou*iou)/sigma)

        # remove intervals with score < threshold
        intervals = rest[rest[:,2] > score_threshold]

    return torch.stack(keep)

def run_inference_new(model, dataloader, device, threshold, iou_threshold, metadata_list):
    """
    Führt Inferenz durch und ordnet jede Vorhersage exakt dem Slice in metadata_list zu.
    Gibt eine Liste von Einträgen zurück:
    {
      "original_idx": int,
      "segment_idx": int,
      "preds": [ { "class": c, "intervals": [[start_col, end_col, score], ...] }, ... ]
    }
    """
    preds_by_slice = []
    slice_idx = 0
    model.eval()

    with torch.no_grad():
        for batch in dataloader:
            # Tensoren auf Device bringen
            for k, v in batch.items():
                if isinstance(v, torch.Tensor):
                    batch[k] = v.to(device, non_blocking=True)

            # Autocast nur auf CUDA aktivieren
            use_autocast = (isinstance(device, str) and device.startswith("cuda")) or (hasattr(device, "type") and device.type == "cuda")
            autocast_ctx = torch.amp.autocast(device_type="cuda", dtype=torch.float16) if use_autocast else contextlib.nullcontext()

            with autocast_ctx:
                class_preds, regr_preds = model(batch["input_features"])
                class_probs = torch.sigmoid(class_preds)

            B, T, C = class_preds.shape
            for b in range(B):
                # passendes Slice aus metadata_list holen
                meta = metadata_list[slice_idx]
                slice_idx += 1

                preds_per_class = []
                for c in range(C):
                    intervals = []
                    for t in range(T):
                        score = class_probs[b, t, c]
                        if float(score) > threshold:
                            start = t - regr_preds[b, t, 0]
                            end   = t + regr_preds[b, t, 1]
                            intervals.append(torch.stack([start, end, score]))

                    if len(intervals) > 0:
                        intervals = torch.stack(intervals)
                        intervals = nms_1d_torch(intervals, iou_threshold=iou_threshold)
                        intervals = intervals.cpu().tolist()
                    else:
                        intervals = []

                    preds_per_class.append({"class": c, "intervals": intervals})

                preds_by_slice.append({
                    "original_idx": meta["original_idx"],
                    "segment_idx": meta["segment_idx"],
                    "preds": preds_per_class
                })

    return preds_by_slice


def reconstruct_predictions(preds_by_slice, total_spec_columns, ID_TO_CLUSTER):
    """
    Rekonstruiert alle Vorhersagen aus Slice-Koordinaten in Datei-Zeitkoordinaten.
    Gibt ein Dict mit Listen zurück: {"onset": [], "offset": [], "cluster": [], "score": []}
    """
    grouped_preds = defaultdict(list)
    for ps in preds_by_slice:
        grouped_preds[ps["original_idx"]].append(ps)

    sec_per_col = 0.02
    cols_per_segment = total_spec_columns // 2  # T entspricht total_spec_columns/2

    all_preds_final = {"onset": [], "offset": [], "cluster": [], "score": [], "orig_idx": []}

    # Über alle Originaldateien iterieren
    for orig_idx in sorted(grouped_preds.keys()):
        segs_sorted = sorted(grouped_preds[orig_idx], key=lambda x: x["segment_idx"])
        for seg in segs_sorted:
            offset_cols = seg["segment_idx"] * cols_per_segment
            for p in seg["preds"]:
                c = p["class"]
                for (start_col, end_col, score) in p["intervals"]:
                    start_sec = (offset_cols + start_col) * sec_per_col
                    end_sec   = (offset_cols + end_col)   * sec_per_col
                    all_preds_final["onset"].append(float(start_sec))
                    all_preds_final["offset"].append(float(end_sec))
                    # Map Klasse-ID -> Cluster-Label
                    #all_preds_final["cluster"].append(ID_TO_CLUSTER[c] if c in range(len(ID_TO_CLUSTER)) else "unknown")
                    all_preds_final["cluster"].append(ID_TO_CLUSTER.get(c, "unknown"))
                    all_preds_final["score"].append(float(score))
                    all_preds_final["orig_idx"].append(orig_idx)

    return all_preds_final

def nms_1d_torch(intervals: torch.Tensor, iou_threshold):
    """
    intervals: Tensor [N, 3] -> (start, end, score)
    iou_threshold: IoU Threshold for suppression

    returns: Tensor [M, 3] of kept intervals
    """
    if intervals.numel() == 0:
        return intervals.new_zeros((0, 3))

    starts = intervals[:, 0]
    ends = intervals[:, 1]
    scores = intervals[:, 2]

    # Sort by score descending
    order = scores.argsort(descending=True)
    keep = []

    while order.numel() > 0:
        i = order[0]
        keep.append(i)

        if order.numel() == 1:
            break

        # compute IoU with the rest
        ss = torch.maximum(starts[i], starts[order[1:]])
        ee = torch.minimum(ends[i], ends[order[1:]])
        inter = torch.clamp(ee - ss, min=0)

        union = (ends[i] - starts[i]) + (ends[order[1:]] - starts[order[1:]]) - inter
        iou = inter / union

        # keep only intervals with IoU <= threshold
        inds = (iou <= iou_threshold).nonzero(as_tuple=False).squeeze(1)
        order = order[inds + 1]
    out = intervals[torch.tensor(keep, dtype=torch.long, device=intervals.device)]
    #out = intervals[keep]
    if out.ndim == 1:   # turn single interval into [1,3]
        out = out.unsqueeze(0)
    return out


def group_by_file(all_preds, all_labels, metadata_list):
    """Gives back dicts grouped by file index: {file_idx: {'onset':[], 'offset':[], 'cluster':[], 'score':[]}}"""
    # group preds
    preds_grouped = defaultdict(lambda: {"onset": [], "offset": [], "cluster": [], "score": []})
    for i, o in enumerate(all_preds["onset"]):
        file_idx = all_preds["orig_idx"][i]  

        preds_grouped[file_idx]["onset"].append(all_preds["onset"][i])
        preds_grouped[file_idx]["offset"].append(all_preds["offset"][i])
        preds_grouped[file_idx]["cluster"].append(all_preds["cluster"][i])
        preds_grouped[file_idx]["score"].append(all_preds["score"][i])

    # group labels
    labels_grouped = defaultdict(lambda: {"onset": [], "offset": [], "cluster": [], "quality": []})
    for i, o in enumerate(all_labels["onset"]):
        file_idx = all_labels["orig_idx"][i]
        labels_grouped[file_idx]["onset"].append(all_labels["onset"][i])
        labels_grouped[file_idx]["offset"].append(all_labels["offset"][i])
        labels_grouped[file_idx]["cluster"].append(all_labels["cluster"][i])
        labels_grouped[file_idx]["quality"].append(all_labels["quality"][i])
    
    return preds_grouped, labels_grouped


def filter_by_class(data_dict, target_class):
    """Filters onset/offset/cluster/score-entries by target_class."""
    mask = [c == target_class for c in data_dict["cluster"]]
    return {
        "onset":  [o for o, m in zip(data_dict["onset"], mask) if m],
        "offset": [o for o, m in zip(data_dict["offset"], mask) if m],
        "cluster": [c for c, m in zip(data_dict["cluster"], mask) if m],
        "score": [s for s, m in zip(data_dict.get("score", [0]*len(mask)), mask) if m]
    }

def filter_by_class_labels(data_dict, target_class):
    """Filters onset/offset/cluster/quality-entries by target_class."""
    mask = [c == target_class for c in data_dict["cluster"]]
    return {
        "onset":  [o for o, m in zip(data_dict["onset"], mask) if m],
        "offset": [o for o, m in zip(data_dict["offset"], mask) if m],
        "cluster": [c for c, m in zip(data_dict["cluster"], mask) if m],
        "quality": [s for s, m in zip(data_dict.get("quality", [0]*len(mask)), mask) if m]
    }


class EarlyStopping:
    def __init__(self, patience=5, min_delta=0.0):
        """
        Args:
            patience (int): How many epochs to wait after last improvement.
            min_delta (float): Minimum change to qualify as improvement.
        """
        self.patience = patience
        self.min_delta = min_delta
        self.best_loss = float("inf")
        self.counter = 0
        self.should_stop = False

    def step(self, current_loss):
        if self.best_loss - current_loss > self.min_delta:
            self.best_loss = current_loss
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.should_stop = True


def evaluate_detection_metrics_with_false_class_qualities(labels, predictions, overlap_tolerance, allowed_qualities = None):

    label_onsets   = labels['onset']
    label_offsets  = labels['offset']
    label_clusters = labels['cluster']
    label_qualities = labels['quality']

    #print(allowed_qualities)


    if str(allowed_qualities) != 'None':
        # try to convert everything to int (if possible)
        #print('Oh Noo')
        try:
            allowed_ints = set(int(q) for q in allowed_qualities)
            label_ints = np.array([int(float(q)) for q in label_qualities])
            mask = np.array([q in allowed_ints for q in label_ints], dtype=bool)
        except ValueError:
            # else use string comparison 
            allowed_str = set(str(q) for q in allowed_qualities)
            qual_str = np.array([str(q) for q in label_qualities])
            mask = np.array([q in allowed_str for q in qual_str], dtype=bool)

        label_onsets   = np.array(label_onsets)[mask]
        label_offsets  = np.array(label_offsets)[mask]
        label_clusters = np.array(label_clusters)[mask]
        label_qualities = np.array(label_qualities)[mask]

    # load predictions
    pred_onsets = np.array(predictions['onset'])
    pred_offsets = np.array(predictions['offset'])
    pred_clusters = np.array(predictions['cluster'])
    pred_scores = np.array(predictions['score'])
    
    # sort predictions by score descending
    order = np.argsort(-pred_scores)
    pred_onsets = pred_onsets[order]
    pred_offsets = pred_offsets[order]
    pred_clusters = pred_clusters[order]
    pred_scores = pred_scores[order]
    

    matched_labels = set()
    matched_preds = set()
    false_class = 0

    gtp = len(label_onsets)     
    pp  = len(pred_onsets)

    # Matching
    for p_idx, (po, pf, pc) in enumerate(zip(pred_onsets, pred_offsets, pred_clusters)):
        for l_idx, (lo, lf, lc) in enumerate(zip(label_onsets, label_offsets, label_clusters)):
            if l_idx in matched_labels or p_idx in matched_preds:
                continue
            inter = max(0.0, min(pf, lf) - max(po, lo))
            union = max(pf, lf) - min(po, lo)
            ov = inter / union if union > 0 else 0.0
            if ov > overlap_tolerance and str(pc) == str(lc):
                matched_labels.add(l_idx)
                matched_preds.add(p_idx)

    for p_idx, (po, pf, pc) in enumerate(zip(pred_onsets, pred_offsets, pred_clusters)):
        for l_idx, (lo, lf, lc) in enumerate(zip(label_onsets, label_offsets, label_clusters)):
            if l_idx in matched_labels or p_idx in matched_preds:
                continue
            inter = max(0.0, min(pf, lf) - max(po, lo))
            union = max(pf, lf) - min(po, lo)
            ov = inter / union if union > 0 else 0.0
            if ov > overlap_tolerance and str(pc) != str(lc):
                false_class +=1
                matched_labels.add(l_idx)
                matched_preds.add(p_idx)


    tp = len(matched_labels) - false_class
    fp = len(pred_onsets) - len(matched_preds)
    fn = len(label_onsets) - len(matched_labels) 
    fc = false_class

    precision = tp / (tp + fp + fc) if (tp + fp + fc) > 0 else 0.0
    recall    = tp / (tp + fn + fc) if (tp + fn + fc) > 0 else 0.0
    f1        = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

    return {
        'gtp': gtp, 'pp': pp,
        'tp': tp, 'fp': fp, 'fn': fn, 'fc': fc,
        'precision': precision, 'recall': recall, 'f1': f1
    }



def losses_val(
        out_cls_logits,          # [B, T/2, C]
        out_offsets,             # [B, T/2, 2]
        gt_cls_labels,           # [B, T/2, C]
        gt_offsets,              # [B, T/2, 2]
        train_loss_weight=1,
        loss_normalizer=200,
        loss_normalizer_momentum=0.8,
        class_weights=None       # <--- NEU: Tensor [C]
    ):
    """
    out_cls_logits: Rohlogits der Klassen [B, T, C]
    gt_cls_labels:  One-Hot Labels [B, T, C]
    class_weights:  Tensor [C], höhere Werte für seltene Klassen
    """

    B, T, C = out_cls_logits.shape
    assert gt_cls_labels.shape == (B, T, C)
    assert out_offsets.shape == (B, T, 2)
    assert gt_offsets.shape == (B, T, 2)

    # positive Positionen (nur dort gibt es Regression)
    pos_mask = gt_offsets.sum(-1) > 0

    pred_offsets_pos = out_offsets[pos_mask]
    gt_offsets_pos = gt_offsets[pos_mask]

    num_pos = pos_mask.sum().item()

    # ---------- Klassen-Loss mit Gewichtung ----------
    if class_weights is not None:
        # class_weights -> [C]  -> [1,1,C] broadcastfähig
        weights = class_weights.view(1, 1, -1).to(out_cls_logits.device)

        # Focal-Loss für alle Klassen (reduction='none')
        raw_loss = sigmoid_focal_loss(out_cls_logits, gt_cls_labels, reduction='none')  # [B,T,C]
        weighted_loss = raw_loss * weights
        cls_loss = weighted_loss.sum() / max(num_pos, 1)
    else:
        # Standard ohne Gewichtung
        cls_loss = sigmoid_focal_loss(out_cls_logits, gt_cls_labels, reduction='sum') / max(num_pos, 1)

    # ---------- Regression ----------
    if num_pos == 0:
        reg_loss = 0.0 * pred_offsets_pos.sum()
    else:
        reg_loss = ctr_diou_loss_1d(pred_offsets_pos, gt_offsets_pos, reduction='sum')
        reg_loss = reg_loss / num_pos

    # ---------- Balancing ----------
    if train_loss_weight > 0:
        loss_weight = train_loss_weight
    else:
        loss_weight = cls_loss.detach() / max(reg_loss.item(), 0.01)

    final_loss = cls_loss + reg_loss * loss_weight
    return cls_loss, reg_loss, final_loss



def load_actionformer_model(initial_model_path, num_classes, num_decoder_layers, num_head_layers, dropout):
    """Load ActionFormer model with Whisper encoder"""
    from transformers import WhisperModel
    
    # Load Whisper encoder #????this should be part of the model already!
    whisper_model = WhisperModel.from_pretrained("openai/whisper-small", local_files_only=True)
    #whisper_model = WhisperModel.from_pretrained("/projects/extern/CIDAS/cidas_digitalisierung_lehre/mthesis_sophie_dierks/dir.project/lemurcalls/lemurcalls/whisper_models/whisper_large")


    encoder = whisper_model.encoder
    
    # Create ActionFormer model with the correct number of classes
    model = WhisperFormer(encoder, num_classes=num_classes, num_decoder_layers=num_decoder_layers, num_head_layers=num_head_layers, dropout=dropout)
    
    # Load pretrained weights if available
    if initial_model_path and os.path.exists(initial_model_path):
        print(f"Loading pretrained weights from {initial_model_path}")
        model.load_state_dict(torch.load(initial_model_path, map_location='cpu'))
    
    return model

# for debugging
def actionformer_train_iteration(model, batch, optimizer, scheduler, scaler, device):
    """
    Training iteration für ActionFormer mit AMP Debugging.
    Prüft Gradienten, Logits, Shapes etc. ohne das Training zu unterbrechen.
    """
    # Batch auf Device
    #for key in batch:
    #    batch[key] = batch[key].to(device)
    for k, v in batch.items():
        if isinstance(v, torch.Tensor):
            batch[k] = v.to(device, non_blocking=True)

    optimizer.zero_grad()
    torch.cuda.empty_cache()

    # Forward in autocast
    #with autocast(dtype=torch.float16, enabled=(device.type == "cuda")):
    with autocast(dtype=torch.float16, enabled=False):
        class_preds, regr_preds = model(batch["input_features"])
        #print(batch['raw_labels'])

        #with torch.no_grad():
        #    probs = torch.sigmoid(class_preds)
        #    print("Mean probability per class:", probs.mean(dim=(0,1)).cpu().numpy())
        # Compute loss
        cls_loss, reg_loss, total_loss = losses_val(
            class_preds,
            regr_preds,
            batch['clusters'],
            batch['segments'],
            loss_normalizer=200.0,
            loss_normalizer_momentum=0.8,
            train_loss_weight=args.train_loss_weight,
            #class_weights=class_weights.to(device),
            class_weights=None
            
        )
    clusters = batch['clusters']
    segments = batch['segments']


    # Backward über GradScaler
    scaler.scale(total_loss).backward()

    # Optimizer step + scaler update
    scaler.step(optimizer)
    scaler.update()

    if args.lr_schedule == "linear":
        scheduler.step()

    return float(cls_loss.detach()), float(reg_loss.detach()), float(total_loss.detach())




def actionformer_validation_loss(model, val_dataloader, device): #NEIN hier anderer loss notwendig!!
    was_training = model.training
    model.eval()

    sum_loss = 0.0
    sum_loss_class = 0.0
    sum_loss_reg = 0.0
    n_batches = 0

    with torch.no_grad():
        for batch in val_dataloader:
            #for k in batch:
            #    batch[k] = batch[k].to(device, non_blocking=True)

            for k, v in batch.items():
                if isinstance(v, torch.Tensor):
                    batch[k] = v.to(device, non_blocking=True)

            class_preds, regr_preds = model(batch["input_features"])
            cls_loss, reg_loss, total = losses_val(
                class_preds, regr_preds, batch["clusters"], batch["segments"],
                train_loss_weight=args.train_loss_weight,
                loss_normalizer=200,
                loss_normalizer_momentum=0.8,
                #class_weights=class_weights.to(device)
                class_weights=None
            )
            # korrekt auf CPU in Python-Float summieren
            sum_loss += float(total.detach().cpu())
            sum_loss_class += float(cls_loss.detach().cpu())
            sum_loss_reg += float(reg_loss.detach().cpu())
            n_batches += 1

    if was_training:
        model.train()

    return (sum_loss / max(n_batches, 1), sum_loss_class / max(n_batches, 1), sum_loss_reg / max(n_batches, 1))



def actionformer_validation_f1_allclasses(model, dataloader, device, iou_threshold, cluster_codebook,total_spec_columns, feature_extractor,
num_classes, low_quality_value, batch_size, num_workers, collate_fn, ID_TO_CLUSTER, overlap_tolerance, allowed_qualities, all_labels,metadata_list ):
    was_training = model.training
    model.eval()

    preds_by_slice = run_inference_new(
    model=model,
    dataloader=dataloader,          # muss mit shuffle=False erstellt sein
    device=device,
    threshold=0,
    iou_threshold=iou_threshold,
    metadata_list=metadata_list     # kommt aus slice_audios_and_labels
    )

    final_preds = reconstruct_predictions(
    preds_by_slice=preds_by_slice,
    total_spec_columns=total_spec_columns,
    ID_TO_CLUSTER=ID_TO_CLUSTER     # aus datautils importiert
    )

    return final_preds

def evaluate(final_preds, threshold, overlap_tolerance, allowed_qualities, all_labels, metadata_list, model ):
    was_training = model.training
    model.eval()
    all_preds_final  = {"onset": [], "offset": [], "cluster": [], "score": [], "orig_idx": []}
    #only preds above threshold
    #print(final_preds["score"])
    if "score" in final_preds:
        filtered = [
            (on, off, cl, sc, oi)
            for on, off, cl, sc, oi in zip(
                final_preds["onset"],
                final_preds["offset"],
                final_preds["cluster"],
                final_preds["score"],
                final_preds["orig_idx"]
            )
            if float(sc) > threshold
        ]
        if len(filtered) == 0:
            final_preds["onset"] = []
            final_preds["offset"] = []
            final_preds["cluster"] = []
            final_preds["score"] = []
            final_preds["orig_idx"] = []
        else:
            (
                final_preds["onset"],
                final_preds["offset"],
                final_preds["cluster"],
                final_preds["score"],
                final_preds["orig_idx"]
            ) = zip(*filtered)
    else:
        print("Vorsicht, keine scores in final_preds")


    all_preds_final["onset"].extend(final_preds["onset"])
    all_preds_final["offset"].extend(final_preds["offset"])
    all_preds_final["cluster"].extend(final_preds["cluster"])
    all_preds_final["score"].extend(final_preds["score"])
    all_preds_final["orig_idx"].extend(final_preds["orig_idx"])

    all_preds, all_labels = group_by_file(all_preds_final, all_labels, metadata_list)
    #print(f'all_labels: {all_labels}')
    tps, fps, fns, fcs, gtps, pps = [],[],[],[],[],[]

    for idx in range(len(all_preds)):

        metrics = evaluate_detection_metrics_with_false_class_qualities(all_labels[idx], all_preds[idx], overlap_tolerance, allowed_qualities = allowed_qualities)
        tps.append(metrics['tp'])
        fps.append(metrics['fp'])
        fns.append(metrics['fn'])
        fcs.append(metrics['fc'])
        gtps.append(metrics['gtp'])
        pps.append(metrics['pp'])
    
    tp_total = sum(tps)
    fp_total = sum(fps)
    fc_total = sum(fcs)
    fn_total = sum(fns)
    gtp_total = sum(gtps)
    pp_total = sum(pps)

    precision = tp_total / (tp_total + fp_total + fc_total) if (tp_total + fp_total + fc_total) > 0 else 0
    recall    = tp_total / (tp_total + fn_total + fc_total) if (tp_total + fn_total + fc_total) > 0 else 0
    f1_all    = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
    if was_training:
        model.train()

    return {
        "f1": f1_all,
        "precision": precision,
        "recall": recall,
        "tp_total": tp_total,
        "gtp_total": gtp_total,
        "pp_total": pp_total      
    }

def compute_class_weights_from_label_list(label_list, codebook):
    """
    label_list: Liste von Dicts mit ['cluster']
    codebook: Mapping {cluster_str -> class_id}
    """
    num_classes = max(codebook.values()) + 1
    counts = np.zeros(num_classes, dtype=np.int64)

    for labels in label_list:
        for cl in labels["cluster"]:       
            if cl in codebook:
                class_id = codebook[cl]
                counts[class_id] += 1
            else:
                print(f"Unbekannter Clustes '{cl}' wird ignoriert.")

    # Gewichte invers zur Häufigkeit
    weights = 1.0 / (counts + 1e-6)
    weights = weights / weights.mean() 

    return torch.tensor(weights, dtype=torch.float32), counts



def collate_fn(batch):

    input_features = [item["input_features"].clone().detach().float() for item in batch]
    segments = torch.stack([item["segments"] for item in batch])
    clusters = torch.stack([item["clusters"] for item in batch])

    input_features = torch.stack(input_features) 

    # Rohlabels einfach in eine Liste legen
    raw_labels = [item.get("raw_labels", None) for item in batch]


    return {
        "input_features": input_features,       # [B, F, T] for B=batch_size, F=feature_dim, T=total_spec_columns
        "segments": segments,                   # [B, T/2 , 2]
        "clusters": clusters,                    # [B, T/2, C]
        "raw_labels": raw_labels             # Liste der Original-Labels (len = B)
    }


if __name__ == "__main__":
    
    parser = argparse.ArgumentParser()

    parser.add_argument("--initial_model_path" )
    parser.add_argument("--train_dataset_folder", default=None )
    parser.add_argument("--model_folder" )
    parser.add_argument("--audio_folder" )
    parser.add_argument("--label_folder" )
    parser.add_argument("--label_folder_val" )
    parser.add_argument("--audio_folder_val" )
    parser.add_argument("--n_device", type = int, default = 1 )
    parser.add_argument("--gpu_list", type = int, nargs = "+", default = None )
    parser.add_argument("--project", default = "wseg-lemur" )
    parser.add_argument("--run_name", default = None )
    parser.add_argument("--run_notes", default = None )
    parser.add_argument("--run_tags", default = None, nargs='+')
    parser.add_argument("--update_every", type = int, default = 100 )
    parser.add_argument("--validate_every", type = int, default = None )
    parser.add_argument("--validate_per_epoch", type = int, default = 0 )
    parser.add_argument("--save_every", type = int, default = None )
    parser.add_argument("--save_per_epoch", type = int, default = 0 )
    parser.add_argument("--max_num_epochs", type = int, default = 40 )
    parser.add_argument("--max_num_iterations", type = int, default = None )
    parser.add_argument("--val_ratio", type = float, default = 0 )
    parser.add_argument("--make_equal", nargs="+", default = None )
    parser.add_argument("--use_early_stopping", action="store_true",                # Flag ohne Wert → True, wenn angegeben
    help="Aktiviere Early Stopping basierend auf Validierungs-F1.")
    parser.add_argument("--num_decoder_layers", type = int, default = 3)
    parser.add_argument("--num_head_layers", type = int, default = 2)


    parser.add_argument("--patience", type = int, default = 8, help="If the validation score does not improve for [patience] epochs, stop training.")
    parser.add_argument("--total_spec_columns", type = int, default = 3000 )
    parser.add_argument("--batch_size", type = int, default = 16 )
    parser.add_argument("--learning_rate", type = float, default = 2e-4)
    parser.add_argument("--lr_schedule", default = 'cosine')
    parser.add_argument("--seed", type = int, default = 66100 )
    parser.add_argument("--weight_decay", type = float, default = 0.001 )
    parser.add_argument("--warmup_steps", type = int, default = 100 )
    parser.add_argument("--freeze_encoder", type = bool, default = True)
    parser.add_argument("--dropout", type = float, default = 0.1 )
    parser.add_argument("--num_workers", type = int, default = 4 )
    parser.add_argument("--num_classes", type = int, default = 3 )
    parser.add_argument("--scheduler_patience", type = int, default = 2)
    parser.add_argument("--factor", type = int, default = 0.3)
    parser.add_argument("--train_loss_weight", type = float, default = 1)
    parser.add_argument("--no_decoder", type = bool, default = False)
    parser.add_argument("--T_max", type = int, default = None)
    parser.add_argument("--eta_min", type = float, default = 1e-6)
    parser.add_argument("--thresholds", type=float, nargs="+", default=[0.4, 0.45, 0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85])   
    parser.add_argument("--iou_threshold", type = float, default = 0.4)
    parser.add_argument("--overlap_tolerance", type=float, default=0.1)
    parser.add_argument("--clear_cluster_codebook", type = int, help="set the pretrained model's cluster_codebook to empty dict. This is used when we train the segmenter on a complete new dataset. Set this to 0 if you just want to slighlt finetune the model with some additional data with the same cluster naming rule.", default = 0 )
    parser.add_argument("--low_quality_value", type=float, default=0.3)
    parser.add_argument("--value_q2", type=float, default=1)
    parser.add_argument("--centerframe_size", type=float, default=0.6)
    parser.add_argument("--allowed_qualities", default=[1,2,3])

    args = parser.parse_args()

    if args.seed is not None:
        np.random.seed(args.seed)  
        
    if args.val_ratio == 0.0 and not args.label_folder_val:
        args.validate_every = None
        args.validate_per_epoch= None

    create_if_not_exists(args.model_folder)

    if args.gpu_list is None:
        args.gpu_list = np.arange(args.n_device).tolist()
        
    device = torch.device(  "cuda:%d"%( args.gpu_list[0] ) if torch.cuda.is_available() else "cpu" )


    model = load_actionformer_model(args.initial_model_path, args.num_classes, args.num_decoder_layers, args.num_head_layers, args.dropout)
    
    if args.freeze_encoder:
        for para in model.encoder.parameters():
            para.requires_grad = False
    else:
        for para in model.encoder.parameters():
            para.requires_grad = True


    if args.audio_folder and args.label_folder:
        audio_path_list_train, label_path_list_train = get_audio_and_label_paths_from_folders(
            args.audio_folder, args.label_folder)
    else:
        audio_path_list_train, label_path_list_train = get_audio_and_label_paths(args.train_dataset_folder)
    
    # Split paths before loading
    if args.val_ratio > 0:
        audio_path_list_train, audio_path_list_val, label_path_list_train, label_path_list_val = train_test_split(
            audio_path_list_train, label_path_list_train, test_size=args.val_ratio, random_state=42)
    else:
        audio_path_list_train, label_path_list_train = audio_path_list_train, label_path_list_train
        audio_path_list_val, label_path_list_val = [], []

    cluster_codebook = FIXED_CLUSTER_CODEBOOK

    audio_list_train, label_list_train = load_data(audio_path_list_train, label_path_list_train, cluster_codebook = cluster_codebook, n_threads = 1 )
    
    #if args.val_ratio > 0:
    #    audio_list_train, audio_list_val, label_list_train, label_list_val = train_test_split(audio_list_train, label_list_train, test_size = args.val_ratio)
    """
    class_weights, counts = compute_class_weights_from_label_list(
        label_list_train,
        FIXED_CLUSTER_CODEBOOK
    )

    print("Class counts:", counts)
    print("Class weights:", class_weights)
    """
    class_weights=None

    #slices audios in chunks of total_spec_columns spectogram columns and adjusts the labels accordingly
    audio_list_train, label_list_train, metadata_list = slice_audios_and_labels( audio_list_train, label_list_train, args.total_spec_columns )
    print(f"Created {len(audio_list_train)} training samples after slicing") 

    feature_extractor = WhisperFeatureExtractor.from_pretrained("openai/whisper-small", local_files_only=True)
    #feature_extractor = WhisperFeatureExtractor.from_pretrained("/projects/extern/CIDAS/cidas_digitalisierung_lehre/mthesis_sophie_dierks/dir.project/lemurcalls/lemurcalls/whisper_models/whisper_large")

    ### Handle Validation Set ###
    if args.val_ratio > 0:
        audio_list_val, label_list_val = load_data(audio_path_list_val, label_path_list_val, cluster_codebook = cluster_codebook, n_threads = 1 )

        audio_list_val, label_list_val, metadata_list_val = slice_audios_and_labels( audio_list_val, label_list_val, args.total_spec_columns )
        print(f"Created {len(audio_list_val)} validation samples after slicing")

        # Create validation dataloader
        val_dataset = WhisperFormerDatasetQuality(audio_list_val, label_list_val, args.total_spec_columns, feature_extractor, args.num_classes, args.low_quality_value, args.value_q2, args.centerframe_size)
        val_dataloader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False,
                                        num_workers=args.num_workers, collate_fn=collate_fn, drop_last=False) 
        
        #---- get labels for calculation of F1 val score ----#
        all_labels = {"onset": [], "offset": [], "cluster": [], "quality": [], "orig_idx": []}
        # Labels laden
        for i, label_path in enumerate(label_path_list_val):
            with open(label_path, "r") as f:
                labels = json.load(f)
            
            clusters = labels["cluster"]
            labels["cluster"] = [ID_TO_CLUSTER[FIXED_CLUSTER_CODEBOOK[c]] for c in clusters]
            

            # Quality-Klassen hinzufügen
            if "quality" in labels:
                quality_list = labels["quality"]
            else:
                quality_list = ["unknown"] * len(labels["onset"])

            # --- globale Sammler befüllen ---
            all_labels["onset"].extend(labels["onset"])
            all_labels["offset"].extend(labels["offset"])
            all_labels["cluster"].extend(labels["cluster"])
            all_labels["quality"].extend(quality_list)
            all_labels["orig_idx"].extend([i for _ in range(len(labels["onset"]))])

    

    if args.label_folder_val:
        audio_path_list_val, label_path_list_val = get_audio_and_label_paths_from_folders(
            args.audio_folder, args.label_folder_val)
    
    if args.label_folder_val:
        audio_path_list_val, label_path_list_val = get_audio_and_label_paths_from_folders(
            args.audio_folder, args.label_folder_val)

        audio_list_val, label_list_val = load_data(audio_path_list_val, label_path_list_val, cluster_codebook = cluster_codebook, n_threads = 1 )

        audio_list_val, label_list_val, metadata_list_val = slice_audios_and_labels( audio_list_val, label_list_val, args.total_spec_columns )
        print(f"Created {len(audio_list_val)} validation samples after slicing")

        # Create validation dataloader
        val_dataset = WhisperFormerDatasetQuality(audio_list_val, label_list_val, args.total_spec_columns, feature_extractor, args.num_classes, args.low_quality_value, args.value_q2, args.centerframe_size)
        val_dataloader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False,
                                        num_workers=args.num_workers, collate_fn=collate_fn, drop_last=False)

        #---- get labels for calculation of F1 val score ----#
        all_labels = {"onset": [], "offset": [], "cluster": [], "quality": [], "orig_idx": []}
        # Labels laden
        for i, label_path in enumerate(label_path_list_val):
            with open(label_path, "r") as f:
                labels = json.load(f)
            
            clusters = labels["cluster"]
            labels["cluster"] = [ID_TO_CLUSTER[FIXED_CLUSTER_CODEBOOK[c]] for c in clusters]
            

            # Quality-Klassen hinzufügen
            if "quality" in labels:
                quality_list = labels["quality"]
            else:
                quality_list = ["unknown"] * len(labels["onset"])

            # --- globale Sammler befüllen ---
            all_labels["onset"].extend(labels["onset"])
            all_labels["offset"].extend(labels["offset"])
            all_labels["cluster"].extend(labels["cluster"])
            all_labels["quality"].extend(quality_list)
            all_labels["orig_idx"].extend([i for _ in range(len(labels["onset"]))])

    # Check if we have any data after slicing
    if len(audio_list_train) == 0:
        print("Error: No valid audio samples after slicing!")
        print("This could be due to:")
        print("  - Audio files that are too short after slicing")
        print("  - No valid segments in the labels")
        print("  - All segments being filtered out during processing")
        sys.exit(1)


    training_dataset = WhisperFormerDatasetQuality(audio_list_train, label_list_train, args.total_spec_columns, feature_extractor, args.num_classes, args.low_quality_value, args.value_q2, args.centerframe_size)

    # Check dataset size before creating DataLoader
    if len(training_dataset) == 0:
        print("Error: Training dataset has 0 samples!")
        sys.exit(1)
    
    if len(training_dataset) < args.batch_size:
        print(f"Warning: Dataset size ({len(training_dataset)}) is smaller than batch size ({args.batch_size})")
        print("Consider reducing batch size or adding more data")

    training_dataloader = DataLoader( training_dataset, batch_size = args.batch_size , shuffle = True , 
                                            worker_init_fn = None, 
                                            num_workers = args.num_workers , drop_last= True,
                                            pin_memory = False,
                                            collate_fn = collate_fn
                                        )

    if len(training_dataloader) == 0:
        print("Error: Too few examples (less than a batch) for training! Exit!")
        sys.exit(1)


    if args.max_num_iterations is not None and args.max_num_iterations > 0:
        args.max_num_epochs = int(np.ceil( args.max_num_iterations / len( training_dataloader )  ))
    else:
        assert args.max_num_epochs is not None and args.max_num_epochs > 0
        args.max_num_iterations = len( training_dataloader ) * args.max_num_epochs
                

    model = nn.DataParallel( model, args.gpu_list )
    model = model.to(device)


    no_decay = ['bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [
        {'params': [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)], 'weight_decay': args.weight_decay},
        {'params': [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
    ]
    
    optimizer = AdamW(optimizer_grouped_parameters, lr = args.learning_rate )

    #initialize the different lr schedulers
    if args.lr_schedule == "linear":
        scheduler = get_linear_schedule_with_warmup(
            optimizer, num_warmup_steps= args.warmup_steps, 
            num_training_steps = args.max_num_iterations
        )
    elif args.lr_schedule == "plateau":
        scheduler = ReduceLROnPlateau(
            optimizer,
            mode="min",               # or "max" if you monitor accuracy/score
            factor=args.factor,               # reduce LR by half
            patience=args.scheduler_patience,               # number of evals with no improvement before reducing
            verbose=True
        )
    elif args.lr_schedule == "cosine":
        T_max = args.T_max if args.T_max is not None else args.max_num_epochs
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=T_max,
            eta_min=args.eta_min
        )
        
    else:
        scheduler = None
    
    scaler = torch.cuda.amp.GradScaler()

    val_score_history = []
    early_stopper = None
    if args.use_early_stopping:
        early_stopper = EarlyStopping(patience=args.patience, min_delta=0.0)

    early_stop = False
    current_step = 0

    train_loss_history = []
    train_loss_history_class=[]
    train_loss_history_reg=[]
    val_loss_history = []
    val_loss_history_class = []
    val_loss_history_reg = []
    lr_reduction_epochs = []
    f1_scores_val = []
    f1_scores_val_m = []
    recall_scores_val = []
    precision_scores_val = []
    best_f1 = 0.0
    best_metrics = {}           

    best_model_state_dict = model.module.state_dict()  # initial save
    #best_model_state_dict = model.state_dict()


    for epoch in range(args.max_num_epochs+1):  
        print(f"\n=== Starting Epoch {epoch} ===")
        model.train() 

        training_losses = []
        training_losses_class = []
        training_losses_reg = []
        val_losses = []
        lrs=[]

        for count, batch in enumerate( tqdm( training_dataloader, desc=f'epoch-{epoch:03}', disable=is_scheduled_job()) ):
            cls_loss, reg_loss, total_loss = actionformer_train_iteration(model, batch, optimizer, scheduler, scaler, device)
            training_losses.append(total_loss)
            training_losses_class.append(cls_loss)
            training_losses_reg.append(reg_loss)

            if count % 100 == 0:
                print(f"Epoch {epoch}, Step {count}, Training Total Loss: {total_loss:.4f}")
                print(f"Epoch {epoch}, Step {count}, Training Class Loss: {cls_loss:.4f}")
                print(f"Epoch {epoch}, Step {count}, Training Regression Loss: {reg_loss:.4f}")
                

        print(f"=== End of Epoch {epoch} ===")
        epoch_train_loss = sum(training_losses)/len(training_losses)
        epoch_train_loss_class = sum(training_losses_class)/len(training_losses_class)
        epoch_train_loss_reg = sum(training_losses_reg)/len(training_losses_reg)
        train_loss_history.append(epoch_train_loss)
        train_loss_history_class.append(epoch_train_loss_class)
        train_loss_history_reg.append(epoch_train_loss_reg)
        print(f"Epoch {epoch}, Step {count}, Epoch Training Loss: {epoch_train_loss:.4f}")
        print(f"val_ratio = {args.val_ratio}, will run validation: {args.val_ratio > 0}")

        
        # Validation at the end of each epoch
        if args.val_ratio > 0 or args.label_folder_val:
            print(f"Running validation for epoch {epoch}...")
            f1s, recalls, precisions = [], [], []
            thresholds = args.thresholds
            final_preds = actionformer_validation_f1_allclasses(model, val_dataloader, device, args.iou_threshold, cluster_codebook, args.total_spec_columns, feature_extractor,
            args.num_classes, args.low_quality_value, args.batch_size, args.num_workers, collate_fn, ID_TO_CLUSTER, args.overlap_tolerance, args.allowed_qualities, all_labels, metadata_list_val)
            for threshold in thresholds:
                results = evaluate(final_preds, threshold, args.overlap_tolerance, args.allowed_qualities, all_labels, metadata_list_val, model)
                pp = results["pp_total"]
                gtp = results["gtp_total"]
                f1 = results["f1"]
                f1s.append(f1)
                recall = results["recall"]
                recalls.append(recall)
                precision = results["precision"]
                precisions.append(precision)

            f1_scores_val.append(f1s)
            recall_scores_val.append(recalls)
            precision_scores_val.append(precisions)
            f1 = np.max(f1s)
            print(f1)
            best_threshold = thresholds[np.argmax(f1s)]
            precision = precisions[np.argmax(f1s)]
            #print(precision)
            recall = recalls[np.argmax(f1s)]
            #print(recall)
            print(f"Epoch {epoch}: Val F1 = {f1:.4f} for threshold {best_threshold}")

            if args.lr_schedule == "cosine":
                if epoch >5:
                    old_lr = get_lr(optimizer)[0]
                    scheduler.step()
                    new_lr = get_lr(optimizer)[0]
                    if new_lr < old_lr:
                        print(f"LR reduced from {old_lr:.2e} to {new_lr:.2e} at epoch {epoch}")
                        lr_reduction_epochs.append((epoch, new_lr))

            elif args.lr_schedule == "plateau":
                old_lr = get_lr(optimizer)[0]
                scheduler.step(1 - f1)
                new_lr = get_lr(optimizer)[0]
                if new_lr < old_lr:
                    print(f"LR reduced from {old_lr:.2e} to {new_lr:.2e} at epoch {epoch}")
                    lr_reduction_epochs.append((epoch, new_lr))



            if f1 > best_f1:
                best_f1 = f1
                best_model_state_dict = copy.deepcopy(model.module.state_dict())
                #best_model_state_dict = copy.deepcopy(model.state_dict())
                best_metrics = {
                "epoch": epoch,
                "f1": f1,
                "precision": precision,
                "recall": recall,
                "threshold": best_threshold
                    }
            
            if early_stopper is not None:
                early_stopper.step(1 - f1)

                if early_stopper.should_stop:
                    print(f"Early stopping triggered at epoch {epoch}. "
                        f"Validation loss did not improve for {args.patience} epochs.")
                    break

        else:
            print(f"No validation set (val_ratio = {args.val_ratio})")

        if current_step >= args.max_num_epochs:
            break

    # Save final model
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    final_model_save_path = f"{args.model_folder}/final_model_{timestamp}"
    
    # Save ActionFormer model
    
    os.makedirs(final_model_save_path, exist_ok=True)

    #plot the losses
        # === Plot Loss Curves ===
    plt.figure(figsize=(8, 5))
    plt.plot(train_loss_history, label="Training Loss")
    plt.plot(train_loss_history_class, label="Training Loss Class")
    plt.plot(train_loss_history_reg, label="Training Loss Regression")
    for i in range(len(thresholds)):
        plt.plot([sublist[i] for sublist in f1_scores_val], label=f"Validation F1 for threshold {thresholds[i]}")
        #plt.plot(f1_scores_val_m, label="Validation F1 Moan")
        #plt.plot([sublist[i] for sublist in recall_scores_val], label=f"Validation Recall for threshold {thresholds[i]}")
        #plt.plot([sublist[i] for sublist in precision_scores_val], label=f"Validation Precision for threshold {thresholds[i]}")
    
    # Vertikale Linien für LR-Reduktionen
    for epoch_idx, lr_val in lr_reduction_epochs:
        plt.axvline(x=epoch_idx, color="k", linestyle="--", alpha=0.7)
        plt.text(epoch_idx, max(train_loss_history), f"LR={lr_val:.1e}",
                rotation=90, verticalalignment="bottom", fontsize=8, color="k")

    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training and Validation Loss")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()

    # Save plot
    loss_plot_path = os.path.join(final_model_save_path, "loss_curve.png")
    plt.savefig(loss_plot_path)
    print("Saved loss curve to:", loss_plot_path)
    plt.close()

    torch.save(best_model_state_dict, f"{final_model_save_path}/best_model.pth")
    print("Best model saved according to early stopping.")


    # Save training arguments
    params_path = os.path.join(final_model_save_path, "training_args.json")
    with open(params_path, "w") as f:
        json.dump(vars(args), f, indent=4)
    
    # Beste Validierungsergebnisse speichern
    metrics_path = os.path.join(final_model_save_path, "best_val_metrics.json")
    with open(metrics_path, "w") as f:
        json.dump(best_metrics, f, indent=4)

    print(f"✅ Best validation metrics saved to {metrics_path}")

    # Pfad zur CSV-Datei
    csv_path = os.path.join(args.model_folder, "runs.csv")

    
    # Zeitstempel im Format YYYY-MM-DD HH:MM:SS
    run_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    # Argumente in ein dict umwandeln (falls argparse.Namespace)
    if hasattr(args, "__dict__"):
        args_dict = vars(args)
    else:
        args_dict = dict(args)

    # Spaltennamen: zuerst "timestamp", dann alle Argument-Namen
    fieldnames = ["timestamp"] + list(args_dict.keys())

    # Prüfen, ob Datei existiert
    file_exists = os.path.isfile(csv_path)

    # CSV schreiben/ergänzen
    with open(csv_path, mode="a", newline="", encoding="utf-8") as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

        # Falls neue Datei → Header schreiben
        if not file_exists:
            writer.writeheader()

        # Zeile mit Zeit + Argumenten schreiben
        row = {"timestamp": run_time}
        row.update(args_dict)
        writer.writerow(row)

    print(f"Run wurde in {csv_path} protokolliert.")

    # Pfad zur CSV-Datei
    csv_path = os.path.join(args.model_folder, "runs.csv")

    # Wenn die Datei noch nicht existiert, Header schreiben
    file_exists = os.path.isfile(csv_path)

    # Öffne CSV im Append-Modus
    with open(csv_path, "a", newline="") as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=list(best_metrics.keys()))
        
        if not file_exists:
            # Header nur schreiben, wenn die Datei neu ist
            writer.writeheader()
        
        # Werte der besten Metrics als neue Zeile eintragen
        writer.writerow(best_metrics)

    print(f"✅ Best metrics also appended to {csv_path}")


        
