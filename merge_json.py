import json
import os
from collections import Counter

def merge_close_calls(data, min_gap=0.1):
    """
    data: dict mit onset[], offset[], cluster[]
    min_gap: maximaler Abstand in Sekunden, damit zwei Calls zusammengeführt werden
    """
    onsets = data["onset"]
    offsets = data["offset"]
    clusters = data["cluster"]

    merged_onsets = []
    merged_offsets = []
    merged_clusters = []

    if not onsets:
        return {"onset": [], "offset": [], "cluster": []}

    # Initialisiere mit erstem Event
    current_onset = onsets[0]
    current_offset = offsets[0]
    current_clusters = [clusters[0]]

    for i in range(1, len(onsets)):
        gap = onsets[i] - current_offset
        if gap <= min_gap:
            # Verschmelzen: erweitere Offset und merke Cluster
            current_offset = max(current_offset, offsets[i])
            current_clusters.append(clusters[i])
        else:
            # Block abschließen
            most_common_cluster = Counter(current_clusters).most_common(1)[0][0]
            merged_onsets.append(current_onset)
            merged_offsets.append(current_offset)
            merged_clusters.append(most_common_cluster)

            # Neues Intervall starten
            current_onset = onsets[i]
            current_offset = offsets[i]
            current_clusters = [clusters[i]]

    # Letzten Block abschließen
    most_common_cluster = Counter(current_clusters).most_common(1)[0][0]
    merged_onsets.append(current_onset)
    merged_offsets.append(current_offset)
    merged_clusters.append(most_common_cluster)

    return {
        "onset": merged_onsets,
        "offset": merged_offsets,
        "cluster": merged_clusters,
    }

if __name__ == "__main__":
    input_path = "/projects/extern/CIDAS/cidas_digitalisierung_lehre/mthesis_sophie_dierks/dir.project/lemurcalls/lemurcalls/model_folder/new_labels/final_model_20250908_142326/inference_results.json"

    # JSON einlesen
    with open(input_path, "r") as f:
        data = json.load(f)

    merged = merge_close_calls(data, min_gap=0.1)

    # Ausgabe im selben Ordner speichern
    folder, filename = os.path.split(input_path)
    name, ext = os.path.splitext(filename)
    output_path = os.path.join(folder, f"{name}_merged{ext}")

    with open(output_path, "w") as f:
        json.dump(merged, f, indent=2)

    print(f"Fertig! Ergebnis gespeichert unter: {output_path}")
