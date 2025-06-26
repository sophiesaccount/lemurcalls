import argparse
import json


def filter_classes(input_path, output_path, delete_classes, key="cluster"):
    with open(input_path, "r", encoding="utf-8") as f:
        daten = json.load(f)
    gefiltert = [eintrag for eintrag in daten if eintrag.get(key) not in delete_classes]
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(gefiltert, f, ensure_ascii=False, indent=2)
    print(f"Gefilterte Datei gespeichert als {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Filtert bestimmte Klassen aus einer JSON-Datei mit Labels.")
    parser.add_argument("--input", required=True, help="Pfad zur Eingabe-JSON-Datei")
    parser.add_argument("--output", required=True, help="Pfad zur Ausgabe-JSON-Datei")
    parser.add_argument("--delete_classes", nargs='+', required=True, help="Zu entfernende Klassen (mehrere durch Leerzeichen getrennt)")
    parser.add_argument("--key", default="cluster", help="Schlüsselname für die Klasse im JSON (default: 'cluster')")
    args = parser.parse_args()
    filter_classes(args.input, args.output, args.delete_classes, args.key)


if __name__ == "__main__":
    main()