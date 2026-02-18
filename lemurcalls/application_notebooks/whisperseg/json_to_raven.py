import json
import pandas as pd
import sys
import os


def json_to_raven_selection_table(json_path, output_path):
    """Convert a predictions JSON file to a Raven selection table (.txt)."""
    with open(json_path, "r") as f:
        predictions = json.load(f)

    required_keys = ["onset", "offset", "cluster"]
    missing = [k for k in required_keys if k not in predictions]
    if missing:
        raise ValueError(f"JSON {json_path} is missing keys: {missing}")

    df = pd.DataFrame({
        "Selection": range(1, len(predictions["onset"]) + 1),
        "View": ["-"] * len(predictions["onset"]),
        "Channel": [1] * len(predictions["onset"]),
        "Begin Time (s)": predictions["onset"],
        "End Time (s)": predictions["offset"],
        "Cluster": predictions["cluster"],
    })

    df.to_csv(output_path, sep="\t", index=False)
    print(f"Raven selection table saved: {output_path}")


def process_folder(input_folder, output_folder):
    """Convert all JSON files in a folder to Raven selection tables."""
    os.makedirs(output_folder, exist_ok=True)

    json_files = [f for f in os.listdir(input_folder) if f.lower().endswith(".json")]

    if not json_files:
        print("No JSON files found in input folder.")
        return

    for json_file in json_files:
        input_path = os.path.join(input_folder, json_file)
        output_file = os.path.splitext(json_file)[0] + ".txt"
        output_path = os.path.join(output_folder, output_file)

        try:
            json_to_raven_selection_table(input_path, output_path)
        except Exception as e:
            print(f"Error processing {json_file}: {e}")


if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python json_to_raven.py <input_folder> <output_folder>")
        sys.exit(1)
    process_folder(sys.argv[1], sys.argv[2])
