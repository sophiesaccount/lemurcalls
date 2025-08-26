import json
import statistics

# Pfad zur JSON-Datei
json_file = "/projects/extern/CIDAS/cidas_digitalisierung_lehre/mthesis_sophie_dierks/dir.project/lemurcalls/lemurcalls/model_folder/new_labels/final_model_20250821_111839/inference_results1.json"

# JSON-Datei einlesen
with open(json_file, 'r') as f:
    data = json.load(f)

# Sicherstellen, dass die Felder existieren
if 'onset' not in data or 'offset' not in data or 'cluster' not in data:
    raise ValueError("Die JSON-Datei muss 'onset', 'offset' und 'cluster' enthalten.")

onsets = data['onset']
offsets = data['offset']

if len(onsets) != len(offsets):
    raise ValueError("Die Längen von 'onset' und 'offset' müssen übereinstimmen.")

# Längen berechnen
lengths = [offset - onset for onset, offset in zip(onsets, offsets) if (offset - onset)>0]

# Statistiken berechnen
average_length = statistics.mean(lengths)
min_length = min(lengths)
max_length = max(lengths)

print(f"Durchschnittliche Länge: {average_length}")
print(f"Minimale Länge: {min_length}")
print(f"Maximale Länge: {max_length}")
