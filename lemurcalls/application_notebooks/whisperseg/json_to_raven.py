import json
import pandas as pd
import sys
import os

def json_to_raven_selection_table(json_path, output_path):
    # Read the predictions JSON
    with open(json_path, "r") as f:
        predictions = json.load(f)
    
    # Quick checks
    required_keys = ["onset", "offset", "cluster"]
    missing = [k for k in required_keys if k not in predictions]
    if missing:
        raise ValueError(f"JSON {json_path} fehlt die Schlüssel: {missing}")

    # Construct DataFrame
    df = pd.DataFrame({
        "Selection": range(1, len(predictions["onset"]) + 1),
        "View": ["-"] * len(predictions["onset"]),
        "Channel": [1] * len(predictions["onset"]),
        "Begin Time (s)": predictions["onset"],
        "End Time (s)": predictions["offset"],
        "Cluster": predictions["cluster"],
        #"Score": predictions["score"]
    })

    # Save as tab-delimited .txt, which Raven expects
    df.to_csv(output_path, sep="\t", index=False)
    print(f"✅ Raven selection table saved: {output_path}")


def process_folder(input_folder, output_folder):
    # Ensure output folder exists
    os.makedirs(output_folder, exist_ok=True)

    # Find all JSON files
    json_files = [f for f in os.listdir(input_folder) if f.lower().endswith(".json")]
    
    if not json_files:
        print("⚠️ Keine JSON-Dateien im Eingabeordner gefunden.")
        return
    
    for json_file in json_files:
        input_path = os.path.join(input_folder, json_file)
        output_file = os.path.splitext(json_file)[0] + ".txt"
        output_path = os.path.join(output_folder, output_file)
        
        try:
            json_to_raven_selection_table(input_path, output_path)
        except Exception as e:
            print(f"❌ Fehler bei {json_file}: {e}")


if __name__ == "__main__":
    # Usage: python script.py input_folder output_folder
    if len(sys.argv) != 3:
        print("Usage: python script.py <input_folder> <output_folder>")
        sys.exit(1)
    
    input_folder = sys.argv[1]
    output_folder = sys.argv[2]

    process_folder(input_folder, output_folder)
