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
from train_copy import collate_fn  # Reuse collate function from training
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


def run_inference(model, dataloader, device):
    dict={}
    dict["cluster"] = []
    dict["time"] = []
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Inference"):
            for key in batch:
                batch[key] = batch[key].to(device)
            
            with torch.amp.autocast(device_type="cuda", dtype=torch.float16):
                class_preds, regr_preds = model(batch["input_features"])
                class_preds = torch.sigmoid(class_preds)
                if class_preds[:,:,0] > 0.5:
                    dict["cluster"].append(class_preds[:,:,0])
                    dict["time"].append(regr_preds[:,:,0    ])
                if class_preds[:,:,1] > 0.5:
                    dict["cluster"].append(class_preds[:,:,1])
                    dict["time"].append(regr_preds[:,:,0])
                else:
                    continue
                    
                
                #preds = torch.sigmoid(class_preds)  # Convert logits to probabilities
                #preds = preds.cpu().numpy()
                #all_class_preds.extend(preds.tolist())
                #all_regr_preds.extend(regr_preds.tolist())
    return dict




#Todo: mit masken arbeiten 

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
                #? hier regr_preds in sekunden, sollten aber eigentlich columns sein??
                # Convert logits to probabilities
                class_probs = torch.sigmoid(class_preds)  
                # Convert logits to probabilities
                #class_probs = class_probs.cpu().numpy()

                #caluclate onsets and offsets, here not working because we do not have columns but seconds!
                (B, T, L) = regr_preds.shape
                _,_,C = class_probs.shape

                #only keep predictions with prob above threshold
                for c in range(C):
                    if class_probs[:,:,c] > threshold:
                        all_class_probs.extend(class_probs.tolist()) #list of length  with entries (B, T, C)=(4, 1500, 2) nicht mehr
                        all_classes.extend(c)
                        all_regr_preds.extend(regr_preds.tolist()) #list of length with entries (B, T, 2)

                        #soft NMS (nms.py nutzen?) wie handle ich die verschiedenen Klassen??
                        #erstmal: nur die n höchten scores behalten?

                #anders speichern? liste unpraktisch? vielleicht hier shcon dictonary?

                #wieder zusammensetzen mit majority voting über die overlaps wie in WhisperSeg

                #dictionary ausgeben



    return all_class_probs, all_regr_preds  





 
        

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
    dict = run_inference_new(model, dataloader, args.device)
    #wieder zusammenfügen 
    #in richtige form bringen!




    # ===== Save results =====
    #with open(args.output_json, "w") as f:
    #    json.dump(dict, f, indent=2)

    #print(f"Inference complete. Results saved to {args.output_json}")
