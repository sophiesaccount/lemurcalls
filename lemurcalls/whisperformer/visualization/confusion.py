import os
import matplotlib.pyplot as plt
from sklearn.metrics import ConfusionMatrixDisplay
import numpy as np

def plot_confusion_matrix(save_dir="."):
    """Plot and save a confusion matrix with predefined values and formatting.

    Args:
        save_dir: Directory to save the figure. Created if missing.
    """
    os.makedirs(save_dir, exist_ok=True)

    # Rows: true label, columns: predicted
    cm_values = np.array([
        [17, 0, 0, 3],
        [0, 4, 0, 1],
        [1, 1, 15, 0],
        [0, 2, 0, 0]
    ])
    display_labels = ["hmm", "moan", "wail", "None"]

    disp = ConfusionMatrixDisplay(confusion_matrix=cm_values, display_labels=display_labels)
    fig, ax = plt.subplots(figsize=(8, 6))
    plot_obj = disp.plot(ax=ax, cmap="Blues", xticks_rotation=45)
    im = plot_obj.im_
    cbar = im.colorbar
    cbar.ax.tick_params(labelsize=16)

    ax.tick_params(axis='both', labelsize=16)
    ax.set_xlabel(ax.get_xlabel(), fontsize=18)
    ax.set_ylabel(ax.get_ylabel(), fontsize=18)
    for text in disp.text_.ravel():
        text.set_fontsize(16)

    plt.tight_layout()
    cm_path = os.path.join(save_dir, "confusion_matrix_classes_none.png")
    plt.savefig(cm_path, dpi=300)
    plt.close()

    print(f"Confusion matrix saved to {cm_path}")

if __name__ == "__main__":
    plot_confusion_matrix(save_dir="/projects/extern/CIDAS/cidas_digitalisierung_lehre/mthesis_sophie_dierks/dir.project/lemurcalls/lemurcalls/CP")
