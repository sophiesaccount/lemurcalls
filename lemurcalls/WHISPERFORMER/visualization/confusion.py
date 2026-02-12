import os
import matplotlib.pyplot as plt
from sklearn.metrics import ConfusionMatrixDisplay
import numpy as np

def plot_confusion_matrix(save_dir="."):
    """
    Plots and saves a confusion matrix with predefined values and formatting.
    """
    # -----------------------------
    # Erstelle das Verzeichnis falls nicht vorhanden
    # -----------------------------
    os.makedirs(save_dir, exist_ok=True)

    # -----------------------------
    # Werte der Confusion Matrix (Zeilen: true, Spalten: predicted)
    # -----------------------------
    cm_values = np.array([
        [17, 0, 0, 3],
        [0, 4, 0, 1],
        [1, 1, 15, 0],
        [0, 2, 0, 0]
    ])

    # Anzeige-Labels
    display_labels = ["hmm", "moan", "wail", "None"]

    # -----------------------------
    # ConfusionMatrixDisplay erstellen
    # -----------------------------
    disp = ConfusionMatrixDisplay(confusion_matrix=cm_values, display_labels=display_labels)
    fig, ax = plt.subplots(figsize=(8, 6))

    # Matrix plotten und Image-Objekt erhalten
    plot_obj = disp.plot(ax=ax, cmap="Blues", xticks_rotation=45)
    im = plot_obj.im_              # Image-Objekt der Matrix
    cbar = im.colorbar             # Colorbar
    cbar.ax.tick_params(labelsize=16)  # Schriftgröße Colorbar

    # Achsenbeschriftung anpassen
    ax.tick_params(axis='both', labelsize=16)
    ax.set_xlabel(ax.get_xlabel(), fontsize=18)
    ax.set_ylabel(ax.get_ylabel(), fontsize=18)

    # Schriftgröße der Zahlen in den Zellen
    for text in disp.text_.ravel():
        text.set_fontsize(16)

    plt.tight_layout()

    # Speichern
    cm_path = os.path.join(save_dir, "confusion_matrix_classes_none.png")
    plt.savefig(cm_path, dpi=300)
    plt.close()

    print(f"Confusion matrix saved to {cm_path}")

# -----------------------------
# Skript ausführen
# -----------------------------
if __name__ == "__main__":
    plot_confusion_matrix(save_dir="/projects/extern/CIDAS/cidas_digitalisierung_lehre/mthesis_sophie_dierks/dir.project/lemurcalls/lemurcalls/CP")
