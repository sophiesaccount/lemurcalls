import json
import argparse
import numpy as np

def evaluate_detection_metrics_with_false_class(labels, predictions, overlap_tolerance=0.0001):
    """Compute TP, FP, FN, FC and F1 for given labels and predictions.

    Args:
        labels: Dict with keys 'onset', 'offset', 'cluster' (lists).
        predictions: Dict with keys 'onset', 'offset', 'cluster' (lists).
        overlap_tolerance: Minimum overlap ratio (0..1) for a match to count. Default 0.0001.

    Returns:
        dict: Keys 'tp', 'fp', 'fn', 'fc', 'f1', 'precision', 'recall'.
    """
    label_onsets = np.array(labels['onset'])
    label_offsets = np.array(labels['offset'])
    label_clusters = np.array(labels['cluster'])

    pred_onsets = np.array(predictions['onset'])
    pred_offsets = np.array(predictions['offset'])
    pred_clusters = np.array(predictions['cluster'])

    matched_labels = set()
    matched_preds = set()
    false_class = 0

    # True Positives and False Class: prediction and label overlap sufficiently
    for p_idx, (po, pf, pc) in enumerate(zip(pred_onsets, pred_offsets, pred_clusters)):
        for l_idx, (lo, lf, lc) in enumerate(zip(label_onsets, label_offsets, label_clusters)):
            if l_idx in matched_labels or p_idx in matched_preds:
                continue
            # Compute overlap
            intersection = max(0, min(pf, lf) - max(po, lo))
            union = max(pf, lf) - min(po, lo)
            overlap_ratio = intersection / union if union > 0 else 0
            if overlap_ratio > overlap_tolerance:
                if str(pc) == str(lc):
                    matched_labels.add(l_idx)
                    matched_preds.add(p_idx)
                else:
                    matched_labels.add(l_idx)
                    matched_preds.add(p_idx)
                    false_class += 1
                break

    tp = len(matched_labels) - false_class
    fp = len(pred_onsets) - len(matched_preds)
    fn = len(label_onsets) - len(matched_labels)
    fc = false_class
    precision = tp / (tp + fp + fc) if (tp + fp + fc) > 0 else 0
    recall    = tp / (tp + fn + fc) if (tp + fn + fc) > 0 else 0
    f1        = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

    return {
        'tp': tp,
        'fp': fp,
        'fn': fn,
        'fc': fc,
        'precision': precision,
        'recall': recall,
        'f1': f1
    }

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Compute detection metrics for labels and predictions.")
    parser.add_argument('--labels', required=True, help='Path to labels JSON file')
    parser.add_argument('--predictions', required=True, help='Path to predictions JSON file')
    parser.add_argument('--overlap_tolerance', type=float, default=0.0001, help='Minimum overlap for a match (default: 0.0001)')
    args = parser.parse_args()

    with open(args.labels, 'r') as f:
        labels = json.load(f)
    with open(args.predictions, 'r') as f:
        predictions = json.load(f)

    metrics = evaluate_detection_metrics_with_false_class(labels, predictions, overlap_tolerance=args.overlap_tolerance)
    print(f"True Positives: {metrics['tp']}")
    print(f"False Positives: {metrics['fp']}")
    print(f"False Negatives: {metrics['fn']}")
    print(f"False Class: {metrics['fc']}")
    print(f"Precision: {metrics['precision']:.4f}")
    print(f"Recall: {metrics['recall']:.4f}")
    print(f"F1-Score: {metrics['f1']:.4f}") 
