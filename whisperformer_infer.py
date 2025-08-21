import argparse
import json
import os
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from whisperformer_dataset import WhisperFormerDataset
from whisperformer_model import WhisperFormer
from transformers import WhisperModel
from datautils import get_audio_and_label_paths_from_folders, load_data, get_cluster_codebook
from datautils import slice_audios_and_labels
from whisperformer_train import collate_fn  # Reuse collate function from training
import numpy as np

def load_trained_whisperformer(checkpoint_path, num_classes, device):
    """Load the WhisperFormer model with Whisper encoder and trained weights."""
    whisper_model = WhisperModel.from_pretrained("openai/whisper-tiny")
    encoder = whisper_model.encoder

    model = WhisperFormer(encoder, num_classes=num_classes)
    model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    model.to(device)
    model.eval()
    return model

def run_inference_new(model, dataloader, device):
    all_class_probs = []
    all_regr_preds = []
    all_classes =[]

    #get predictions for each item from the dataloader
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Inference"):
            for key in batch:
                batch[key] = batch[key].to(device)
            
            with torch.amp.autocast(device_type="cuda", dtype=torch.float16):
                threshold = 0.5
                class_preds, regr_preds = model(batch["input_features"])

                # Convert logits to probabilities
                class_probs = torch.sigmoid(class_preds)  
                #class_probs = class_probs.cpu().numpy()

                #caluclate onsets and offsets, here not working because we do not have columns but seconds!
                (B, T, L) = regr_preds.shape
                _,_,C = class_probs.shape

                #only keep predictions with prob above threshold
                for b in range(B):
                    for c in range(C):
                        pred_list=[]
                        for t in range(T):
                            if class_probs[b,t,c] > threshold: 
                                pred_list.extend( t - regr_preds[b,t,c,0], t + regr_preds[b,t,c,1]) #list 

                        #soft NMS (nms.py nutzen?) wie handle ich die verschiedenen Klassen??
                        #erstmal: nur die n höchten scores behalten?

                #anders speichern? liste unpraktisch? vielleicht hier shcon dictonary?

                #wieder zusammensetzen mit majority voting über die overlaps wie in WhisperSeg

                #dictionary ausgeben

    return all_class_probs, all_regr_preds  


def nms_1d_torch(intervals: torch.Tensor, iou_threshold: float = 0.5):
    """
    intervals: Tensor [N, 3] -> (start, end, score)
              start, end, score können auf GPU liegen
    iou_threshold: IoU Threshold für Suppression

    returns: Tensor [M, 3] der behaltenen Intervalle
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

    return intervals[keep]



def run_inference_new(model, dataloader, device):
    threshold = 0.5
    iou_threshold = 0.5
    all_preds = []

    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Inference"):
            for key in batch:
                batch[key] = batch[key].to(device)
            
            with torch.amp.autocast(device_type="cuda", dtype=torch.float16):
                class_preds, regr_preds = model(batch["input_features"])
                class_probs = torch.sigmoid(class_preds)

                B, T, _ = regr_preds.shape[:3]
                _, _, C = class_probs.shape

                for b in range(B):
                    for c in range(C):
                        intervals = []
                        for t in range(T):
                            score = class_probs[b, t, c]
                            if score > threshold:
                                start = t - regr_preds[b, t, c, 0]
                                end   = t + regr_preds[b, t, c, 1]
                                interval = torch.stack([start, end, score])
                                interval = torch.round(interval * 100) / 100
                                intervals.append(interval)

                        if len(intervals) > 0:
                            intervals = torch.stack(intervals)  # [N, 3]
                            kept = nms_1d_torch(intervals, iou_threshold=iou_threshold)
                        else:
                            kept = torch.empty((0, 3), device=device)

                        all_preds.append({
                            "batch": b,
                            "class": c,
                            "intervals": kept
                        })
    return all_preds




 
        

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint_path", required=True, help="Path to the .pth trained model")
    parser.add_argument("--audio_folder", required=True)
    parser.add_argument("--label_folder", required=True)  # Optional if you have labels for scoring
    parser.add_argument("--output_json", default="inference_results.json")
    parser.add_argument("--max_length", type=int, default=100)
    parser.add_argument("--total_spec_columns", type=int, default=3000)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--num_classes", type=int, default=2)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    # ===== Data loading =====
    audio_paths, label_paths = get_audio_and_label_paths_from_folders(args.audio_folder, args.label_folder)
    
    # Create a dummy cluster codebook
    cluster_codebook = get_cluster_codebook(label_paths, {})
    
    # Load audio + labels
    audio_list, label_list = load_data(audio_paths, label_paths, cluster_codebook=cluster_codebook, n_threads=4)
    
    # Slice to fit model spec length
    audio_list, label_list = slice_audios_and_labels(audio_list, label_list, args.total_spec_columns)

    dataset = WhisperFormerDataset(audio_list, label_list, tokenizer=None, 
                                   max_length=args.max_length, total_spec_columns=args.total_spec_columns)
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, 
                            collate_fn=collate_fn, drop_last=False)

    # ===== Model loading =====
    model = load_trained_whisperformer(args.checkpoint_path, args.num_classes, args.device)

    # ===== Inference =====
    all_preds = run_inference_new(model, dataloader, args.device)
    print(all_preds)

    #wieder zusammenfügen 


    #in richtige form bringen!




    # ===== Save results =====
    #with open(args.output_json, "w") as f:
    #    json.dump(dict, f, indent=2)

    #print(f"Inference complete. Results saved to {args.output_json}")
