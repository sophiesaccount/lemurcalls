import json
import os
import threading
from copy import deepcopy
import sys

import librosa
import numpy as np
from torch.utils.data import Dataset
from tqdm import tqdm

from .audio_utils import WhisperSegFeatureExtractor
from transformers import WhisperFeatureExtractor
from .util.common import is_scheduled_job
from .utils import RATIO_DECODING_TIME_STEP_TO_SPEC_TIME_STEP

import torch
from copy import deepcopy
from collections import Counter


def get_audio_and_label_paths( folder ):
    #wav_list = [ folder + "/" + fname for fname in os.listdir( folder ) if fname.endswith(".wav") ]
    wav_list = [ folder + "/" + fname for fname in os.listdir( folder ) if fname.endswith((".WAV", ".wav")) ]
    audio_paths = []
    label_paths = []
    for wav_name in tqdm(wav_list, desc="prep_data", disable=is_scheduled_job()):
        label_name = wav_name[:-4] + ".json"
        if os.path.exists(label_name):
            audio_paths.append( wav_name )
            label_paths.append( label_name )
    
    return audio_paths, label_paths

def get_audio_and_label_paths_from_folders(audio_folder, label_folder):
    audio_files = {os.path.splitext(f)[0]: os.path.join(audio_folder, f)
                   #for f in os.listdir(audio_folder) if f.endswith(".wav")}
                   for f in os.listdir(audio_folder) if f.endswith((".WAV", ".wav"))}
    label_files = {os.path.splitext(f)[0]: os.path.join(label_folder, f)
                   for f in os.listdir(label_folder) if f.endswith(".json")}
    # Only keep pairs where both audio and label exist
    common_keys = set(audio_files.keys()) & set(label_files.keys())
    audio_paths = [audio_files[k] for k in sorted(common_keys)]

    label_paths = [label_files[k] for k in sorted(common_keys)]
    return audio_paths, label_paths

def get_audio_and_label_paths_from_folders(audio_folder, label_folder):
    audio_files = [f for f in os.listdir(audio_folder) if f.endswith((".WAV", ".wav"))]
    label_files = [f for f in os.listdir(label_folder) if f.endswith((".json", ".jsonr"))]

    audio_paths, label_paths = [], []
    for label in label_files:
        #key = os.path.splitext(label)[0]  # "LEMUR123"
        key = os.path.splitext(label)[0].replace("_preds", "")
        # suche Audio, das diesen Key enthält
        match = [a for a in audio_files if key in a]
        if match:
            audio_paths.append(os.path.join(audio_folder, match[0]))
            label_paths.append(os.path.join(label_folder, label))

    return audio_paths, label_paths

def get_cluster_codebook( label_paths, initial_cluster_codebook ):
    cluster_codebook = deepcopy( initial_cluster_codebook )
    
    unique_clusters = []
    for label_file in label_paths:
        label = json.load( open(label_file) )
        unique_clusters += [ str(cluster) for cluster in label["cluster"]   ]
            
    unique_clusters = sorted(list(set(unique_clusters)))
    
    for cluster in unique_clusters:
        if cluster not in cluster_codebook:
            cluster_codebook[cluster] = len(cluster_codebook)
    return cluster_codebook



def get_cluster_codebook(label_paths, initial_cluster_codebook, make_equal=None):
    """
    Erzeugt ein Cluster-Codebook aus gegebenen Label-Dateien.
    
    Parameter:
    ----------
    label_paths : list[str]
        Liste der Pfade zu JSON-Label-Dateien. Jede Datei enthält 'cluster'.
    initial_cluster_codebook : dict
        Bereits existierendes Codebook {cluster_name: class_id}.
    make_equal : None | list[str] | 'all', optional
        - None: Standardverhalten (keine Änderungen)
        - list[str]: Alle Cluster in dieser Liste werden auf die gleiche Klasse gemappt.
        - 'all': Alle Cluster werden auf die gleiche Klasse gemappt.
    """
    cluster_codebook = deepcopy(initial_cluster_codebook)

    # --- Alle Cluster sammeln ---
    all_clusters = []
    for label_file in label_paths:
        with open(label_file, 'r') as f:
            label = json.load(f)
        all_clusters += [str(cluster) for cluster in label.get("cluster", [])]


    # --- Alle Cluster aus den Label-Dateien sammeln ---
    unique_clusters = []
    for label_file in label_paths:
        with open(label_file, 'r') as f:
            label = json.load(f)
        unique_clusters += [str(cluster) for cluster in label.get("cluster", [])]

    unique_clusters = sorted(list(set(unique_clusters)))

    # --- Häufigkeiten zählen ---
    top_k = 10
    cluster_counts = Counter(all_clusters)
    print(f"\n📊 Top 10 häufigste Cluster:")
    for cluster, count in cluster_counts.most_common(top_k):
        print(f"   {cluster:20s} -> {count} Vorkommen")

    # --- make_equal verarbeiten ---
    if make_equal == 'all':
        # Alle Cluster sollen dieselbe Klasse teilen
        target_class_id = cluster_codebook.get("__merged__", len(cluster_codebook))
        cluster_codebook["__merged__"] = target_class_id
        for cluster in unique_clusters:
            cluster_codebook[cluster] = target_class_id

    elif isinstance(make_equal, (list, set, tuple)):
        # Cluster aus der make_equal-Liste auf eine Klasse mappen
        make_equal = [str(c) for c in make_equal]
        target_class_id = None

        # Falls einer der Namen schon im Codebook existiert, diese ID benutzen
        for c in make_equal:
            if c in cluster_codebook:
                target_class_id = cluster_codebook[c]
                break

        # Falls noch nicht vorhanden, neue Klasse anlegen
        if target_class_id is None:
            target_class_id = len(cluster_codebook)
            cluster_codebook[make_equal[0]] = target_class_id

        # Alle Cluster aus make_equal auf dieselbe Klasse mappen
        for c in make_equal:
            cluster_codebook[c] = target_class_id

        # Alle übrigen Cluster normal hinzufügen
        for cluster in unique_clusters:
            if cluster not in cluster_codebook:
                cluster_codebook[cluster] = len(cluster_codebook)

    else:
        # Standardfall: alles normal hinzufügen
        for cluster in unique_clusters:
            if cluster not in cluster_codebook:
                cluster_codebook[cluster] = len(cluster_codebook)

    # --- Anzahl der Klassen ausgeben ---
    num_classes_new = len(set(cluster_codebook.values()))

    print(f"Number of Classes in Codebook: {num_classes_new}")


    return cluster_codebook

FIXED_CLUSTER_CODEBOOK = {
    "m": 0,   
    "t": 1,   
    "w": 2,   
    "lt": 1,
    "h":1
}
"""
FIXED_CLUSTER_CODEBOOK = {
    "m": 0,   
    "t": 1,   
    "w": 0,   
    "lt": 1,
    "o":1,
    "h":1
}


FIXED_CLUSTER_CODEBOOK = {
    "m": 0,   
    "t": 0,   
    "w": 0,   
    "lt": 0,
    "o":0,
    "h":0
}

FIXED_CLUSTER_CODEBOOK = {
    "vocal": 0,
    "target": 1
}



FIXED_CLUSTER_CODEBOOK = {
    "vocal": 0,
    "mo": 0
}
"""




#ID_TO_CLUSTER = {v: k for k, v in FIXED_CLUSTER_CODEBOOK.items()}

# Optional: inverse Mapping für spätere Rekonstruktion
#ID_TO_CLUSTER = {v: k for k, v in FIXED_CLUSTER_CODEBOOK.items()}


ID_TO_CLUSTER  = {
    0: "m", 
    1: "h",   
    2: "w",   
}

def load_audio_and_label( audio_path_list, label_path_list, thread_id, audio_dict, label_dict, cluster_codebook ):
    local_audio_list = []
    local_label_list = []
    
    for audio_path, label_path in tqdm(zip(audio_path_list, label_path_list), desc="load_data", disable=is_scheduled_job(), leave=False):
        label = json.load(open( label_path ))
        y, _ = librosa.load( audio_path, sr = 16000 )
        y = y.astype(np.float32)
        label["sr"] = 16000
                
        local_audio_list.append( y )

        onset_arr = np.array( label["onset"] )
        offset_arr = np.array( label["offset"] )
        valid_indices = np.logical_and( np.logical_and(  onset_arr < len(y)/label["sr"], offset_arr > 0 ),
                                        onset_arr < offset_arr )
        onset_arr = onset_arr[valid_indices]
        offset_arr = offset_arr[valid_indices]
        onset_arr[ onset_arr < 0 ] = 0
        offset_arr[ offset_arr > len(y)/label["sr"] ] = len(y)/label["sr"]

        label["cluster"] = [ label["cluster"][idx] for idx in np.argwhere(valid_indices)[:,0] ]  
        # Handle quality (optional)
        if "quality" in label:
            label["quality"] = [label["quality"][idx] for idx in np.argwhere(valid_indices)[:, 0]]
        else:
            # Fill with 'unknown' for each valid entry
            label["quality"] = ['unknown'] * len(label["cluster"])
        #label["quality"] = [ label["quality"][idx] for idx in np.argwhere(valid_indices)[:,0] ]       
        cluster_id_arr = np.array( [ cluster_codebook[ str(value) ] for value in label["cluster"] ]  )
        quality_id_arr = np.array(label["quality"])
        
        label.update( {
            "onset":onset_arr,
            "offset":offset_arr,
            "cluster_id":cluster_id_arr,
            "quality": quality_id_arr
        } )
        local_label_list.append( label )

    audio_dict[thread_id] = local_audio_list
    label_dict[thread_id] = local_label_list

def load_data(audio_path_list, label_path_list, cluster_codebook = None, n_threads = 1 ):
    samples_per_thread = int(np.ceil( len(audio_path_list) / n_threads ))
    audio_dict = {}
    label_dict = {}
    thread_list = []
    
    for thread_id, offset in enumerate(range( 0, len(audio_path_list), samples_per_thread )):
        t = threading.Thread( target=load_audio_and_label, args=( audio_path_list[offset:offset+samples_per_thread], 
                                                          label_path_list[offset:offset+samples_per_thread],
                                                          thread_id,
                                                          audio_dict, label_dict,
                                                          cluster_codebook
                                                        ) )
        t.start()
        thread_list.append(t)
    for t in thread_list:
        t.join()
        
    audio_list = []
    label_list = []
    for thread_id in sorted(audio_dict.keys()):
        audio_list += audio_dict[thread_id]
        label_list += label_dict[thread_id]
    
    assert len(audio_list) == len(label_list) 
    
    return audio_list, label_list

def split_audio_and_label( audio, label, split_ratio ):
    """
    Splits an audio signal and its labels into two parts based on a given ratio, 
    adjusting event times and discarding segments shorter than 0.1 seconds.
    """

    num_samples_in_audio = len(audio)
    split_point = int( num_samples_in_audio * split_ratio )
    split_time = split_point / label["sr"] 
    
    audio_part1 = audio[ :split_point ]
    intersected_indices_part1 = label["onset"] < split_time
    label_part1 = deepcopy( label )
    label_part1.update(
    {
        "onset":label["onset"][intersected_indices_part1],
        "offset": np.minimum(label["offset"][intersected_indices_part1], split_time ),
        "cluster_id":label["cluster_id"][intersected_indices_part1],
        "cluster": [ label["cluster"][idx] for idx in np.argwhere( intersected_indices_part1 )[:,0]  ]
    })
    ## drop too short audios
    if len(audio_part1) / label["sr"] < 0.1:
        audio_part1 = None
        label_part1 = None
    
    
    audio_part2 = audio[ split_point: ]
    intersected_indices_part2 = label["offset"] > split_time
    label_part2 = deepcopy( label )
    label_part2.update(
    {
        "onset": np.maximum(label["onset"][intersected_indices_part2], split_time ) - split_time,
        "offset": label["offset"][intersected_indices_part2] - split_time,
        "cluster_id":label["cluster_id"][intersected_indices_part2],
        "cluster": [ label["cluster"][idx] for idx in np.argwhere( intersected_indices_part2 )[:,0] ]
    })

    ## drop too short audios
    if len(audio_part2) / label["sr"] < 0.1:
        audio_part2 = None
        label_part2 = None
    
    return ( audio_part1, label_part1 ), ( audio_part2, label_part2 )

def split_audio_and_label( audio, label, split_ratio ):

    
    audio
    label_new = deepcopy( label )
    label_new.update(
    {
        "onset":label["onset"],
        "offset": np.minimum(label["offset"], split_time ),
        "cluster_id":label["cluster_id"][intersected_indices_part1],
        "cluster": [ label["cluster"][idx] for idx in np.argwhere( intersected_indices_part1 )[:,0]  ],
        "quality": [ label["quality"][idx] for idx in np.argwhere( intersected_indices_part1 )[:,0]  ]

    })
    ## drop too short audios
    if len(audio_part1) / label["sr"] < 0.1:
        audio_part1 = None
        label_part1 = None
    
    return ( audio_part1, label_part1 )


def train_val_split( audio_list, label_list, val_ratio ):
    
    audio_list_train = []
    label_list_train = []
    audio_list_val = []
    label_list_val = []
    
    for audio, label in zip( audio_list, label_list ):
        mode = np.random.choice([0,1])
        if mode == 0:
            (audio_val, label_val), (audio_train, label_train) = split_audio_and_label( audio, label, val_ratio )
        else:
            (audio_train, label_train), (audio_val, label_val) = split_audio_and_label( audio, label, 1-val_ratio )
        
        if audio_train is not None:
            audio_list_train.append( audio_train )
            label_list_train.append( label_train )
        
        if audio_val is not None:
            audio_list_val.append( audio_val )
            label_list_val.append( label_val )
    
    return (audio_list_train, label_list_train), ( audio_list_val, label_list_val )





#for majority voting
def make_trials(audio_list, label_list, total_spec_columns, num_trials=3):
    """Erzeuge mehrere überlappende Trials (z. B. mit 1/3 versetzt)."""
    all_audio, all_label, all_meta = [], [], []
    hop = total_spec_columns // num_trials
    for trial in range(num_trials):
        audios, labels, metas = slice_audios_and_labels(
            audio_list, label_list, total_spec_columns, offset=trial*hop
        )
        # Jede Meta-Info bekommt die Trial-ID
        for m in metas:
            m["trial"] = trial
        all_audio.extend(audios)
        all_label.extend(labels)
        all_meta.extend(metas)
    return all_audio, all_label, all_meta



###vorsicht: ohen left padding!!!!!!
def slice_audios_and_labels(audio_list, label_list, total_spec_columns, num_trials=1):
    """
    Slice audios into overlapping segments with different offsets (0, 1/3, 2/3).
    Returns expanded audio_list, label_list, metadata_list with offset_frac.
    """

    new_audios, new_labels, new_metadata = [], [], []
    sec_per_col = 0.01
    spec_time_step = 0.01
    total_spec_columns=3000

    for orig_idx, (audio, label) in enumerate(zip(audio_list, label_list)):
        sr = 16000
        clip_duration = total_spec_columns * spec_time_step
        num_samples_in_clip = int(round(clip_duration * sr))

        for trial in range(num_trials):
            # Versatz in Samples (0, 1/3, 2/3 des Clip)
            frac = trial / num_trials
            offset_samples = int(round(frac * num_samples_in_clip))

            start = offset_samples
            seg_idx = 0

            while start < len(audio):
                end = start + num_samples_in_clip
                audio_clip = audio[start:end]

                #if len(audio_clip) < sr * 0.1:  # skip super short
                #    break


                # Labels anpassen: nur Events im Zeitfenster behalten
                start_time = start / sr
                end_time = end / sr
                intersected_indices = np.logical_and(
                    label["onset"] < end_time, label["offset"] > start_time
                )

                label_clip = deepcopy(label)
                label_clip.update({
                    "onset": np.maximum(label["onset"][intersected_indices], start_time) - start_time,
                    "offset": np.minimum(label["offset"][intersected_indices], end_time) - start_time,
                    "cluster_id": label["cluster_id"][intersected_indices],
                    "cluster": [label["cluster"][idx] for idx in np.argwhere(intersected_indices)[:, 0]],
                    "quality": [label["quality"][idx] for idx in np.argwhere(intersected_indices)[:, 0]]
                })

                # speichern
                new_audios.append(audio_clip)
                new_labels.append(label_clip)
                new_metadata.append({
                    "original_idx": orig_idx,
                    "segment_idx": seg_idx,
                    "offset_frac": frac,   # tells us in which trial we are, by giving us the offset
                    "trial_id": trial
                })

                start += num_samples_in_clip
                seg_idx += 1


    return new_audios, new_labels, new_metadata

###vorsicht: ohen left padding!!!!!!
def slice_audios_and_labels(audio_list, label_list, total_spec_columns, offset=0):
    """
    Slice audios into overlapping segments with different offsets (0, 1/3, 2/3).
    Returns expanded audio_list, label_list, metadata_list with offset_frac.
    """

    new_audios, new_labels, new_metadata = [], [], []
    sec_per_col = 0.01
    spec_time_step = 0.01
    total_spec_columns=3000

    for orig_idx, (audio, label) in enumerate(zip(audio_list, label_list)):
        sr = 16000
        clip_duration = total_spec_columns * spec_time_step
        num_samples_in_clip = int(round(clip_duration * sr))

        start = offset
        seg_idx = 0

        while start < len(audio):
            end = start + num_samples_in_clip
            audio_clip = audio[start:end]

            #if len(audio_clip) < sr * 0.1:  # skip super short
            #    break


            # Labels anpassen: nur Events im Zeitfenster behalten
            start_time = start / sr
            end_time = end / sr
            intersected_indices = np.logical_and(
                label["onset"] < end_time, label["offset"] > start_time
            )

            label_clip = deepcopy(label)
            label_clip.update({
                "onset": np.maximum(label["onset"][intersected_indices], start_time) - start_time,
                "offset": np.minimum(label["offset"][intersected_indices], end_time) - start_time,
                "cluster_id": label["cluster_id"][intersected_indices],
                "cluster": [label["cluster"][idx] for idx in np.argwhere(intersected_indices)[:, 0]],
                "quality": [label["quality"][idx] for idx in np.argwhere(intersected_indices)[:, 0]]
            })

            # speichern
            new_audios.append(audio_clip)
            new_labels.append(label_clip)
            new_metadata.append({
                "original_idx": orig_idx,
                "segment_idx": seg_idx,
                "offset_frac": offset,   # tells us in which trial we are, by giving us the offset
                "trial_id": offset
            })

            start += num_samples_in_clip
            seg_idx += 1


    return new_audios, new_labels, new_metadata


#for WhisperSeg not WhisperFormer
class VocalSegDataset(Dataset):
    def __init__(self, audio_list, label_list, tokenizer, max_length, total_spec_columns, species_codebook ):
        self.audio_list = audio_list
        self.label_list = label_list
        self.feature_extractor_bank = self.get_feature_extractor_bank( label_list )
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.total_spec_columns = total_spec_columns
        self.species_codebook = species_codebook
        
    def get_feature_extractor_bank(self, label_list ):
        feature_extractor_bank = {}
        for label in label_list:
            key = "%s-%s-%s"%( str( label["sr"] ), str(label["spec_time_step"]), str(label["min_frequency"]) )
            if key not in feature_extractor_bank:
                feature_extractor_bank[key] = WhisperFeatureExtractor( label["sr"], label["spec_time_step"], label["min_frequency"] )
        return feature_extractor_bank
        
    def map_time_to_spec_col_index(self, t, spec_time_step ):
        return min( int(np.round( t/( spec_time_step * RATIO_DECODING_TIME_STEP_TO_SPEC_TIME_STEP ) )), self.total_spec_columns  )
        
    def __len__(self):
        return len(self.audio_list)
    
    def __getitem__(self, idx ):
        
        audio = self.audio_list[idx]
        label = self.label_list[idx]

        sr = label["sr"]
        spec_time_step = label["spec_time_step"]
        min_frequency = label["min_frequency"]
        feature_extractor = self.feature_extractor_bank[ "%s-%s-%s"%( str(sr), str(spec_time_step), str(min_frequency) ) ]
        
        num_samples_in_clip = int(np.round( self.total_spec_columns * spec_time_step * sr ))
                    
        clip_start = np.random.choice( min( num_samples_in_clip+1, len(audio) - feature_extractor.n_fft + 1 ) )        
        audio_clip = audio[ clip_start: clip_start + num_samples_in_clip ]
        
        actual_clip_duration = len( audio_clip ) / sr
        start_time = clip_start / sr
        end_time = start_time + actual_clip_duration
        
        intersected_indices = np.logical_and( label["onset"] < end_time, label["offset"] > start_time )
        
        onset_in_clip = np.maximum( label["onset"][intersected_indices], start_time ) - start_time 
        offset_in_clip = np.minimum( label["offset"][intersected_indices], end_time ) - start_time
        cluster_id_in_clip = label["cluster_id"][intersected_indices]
        
        """
        The following code part convert the onset, offset, and cluster_id array into label texts
        onset_timestamp + cluster_id + offset_timestamp: e.g.,
        <|zebra_finch|><|0|>7<|6|><|16|>6<|18|>
        """
        label_text = [ self.species_codebook.get( label["species"], "<|unknown|>" )  ]
        
        for pos in range(len(onset_in_clip)):
            label_text.append( "<|%d|>%d<|%d|>"%(
                                    self.map_time_to_spec_col_index( onset_in_clip[pos], spec_time_step ),
                                    cluster_id_in_clip[pos],
                                    self.map_time_to_spec_col_index( offset_in_clip[pos], spec_time_step ),
                                )
                             )
        label_text = "".join( label_text )
        
        audio_clip = np.concatenate( [ audio_clip, np.zeros( num_samples_in_clip - len(audio_clip) ) ], axis = 0 ).astype(np.float32)
        input_features = feature_extractor(audio_clip, sampling_rate = sr, padding = "do_not_pad")["input_features"][0]
        
        decoder_input_ids = self.tokenizer.encode( label_text,  max_length = self.max_length + 1, truncation=True, padding = True )
        labels = decoder_input_ids[1:]
        decoder_input_ids = decoder_input_ids[:-1]
        decoder_input_ids += [ self.tokenizer.pad_token_id ] * ( self.max_length - len( decoder_input_ids ) )
        labels += [-100] * ( self.max_length  - len(labels) )

        output = {
            "input_features": torch.tensor(input_features, dtype=torch.float32),
            "decoder_input_ids": torch.tensor(np.array(decoder_input_ids), dtype = torch.int64),
            "labels": torch.tensor(np.array(labels), dtype = torch.int64)
        }
        return output






