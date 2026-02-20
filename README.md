# lemurcalls

Automated segmentation and classification of lemur vocalizations using two complementary approaches: **WhisperSeg** (sequence-to-sequence) and **WhisperFormer** (directly predict call centers and regress onsets and offsets).

## Project Structure

```
lemurcalls/
├── lemurcalls/              # Main package
│   ├── whisperseg/          # WhisperSeg: seq2seq segmentation
│   │   ├── model.py         # WhisperSegmenter, WhisperSegmenterFast
│   │   ├── train.py         # Training script
│   │   ├── infer.py         # Single-file inference
│   │   ├── infer_folder.py  # Batch inference over folders
│   │   ├── evaluate.py      # Evaluation metrics
│   │   └── ...
│   ├── whisperformer/       # WhisperFormer: dense prediction segmentation
│   │   ├── model.py         # WhisperFormer model
│   │   ├── train.py         # Training script
│   │   ├── infer.py         # Inference script
│   │   ├── dataset.py       # Dataset classes
│   │   ├── postprocessing/  # Post-processing & filtering
│   │   └── visualization/   # Scatter plots, confusion matrices
│   ├── util/                # Shared utilities (data processing, annotation tools)
│   ├── datautils.py             # Audio/label loading and slicing
│   ├── visualize_predictions.py # Plot spectrograms, scores, GT & predictions
│   ├── convert_hf_to_ct2.py    # HuggingFace to CTranslate2 conversion
│   └── ...
├── tests/                   # Unit tests
├── pyproject.toml           # Package configuration & dependencies
└── README.md
```

## Getting Started

### 1. Clone and create environment

```bash
git clone https://github.com/sophiesaccount/lemurcalls.git
cd lemurcalls

# Create a Python environment (conda/micromamba or venv)
python -m venv .venv

# And activate
.\.venv\Scripts\Activate

```

### 2. Install the package

```bash
pip install -e .
```

This installs `lemurcalls` in editable mode with all dependencies from `pyproject.toml`.

### 3. Download Whisper model weights
!!! Not necessary if you only need inference !!!
The pretrained Whisper encoder weights are not included in the repository (because they are too large). Download them into the `whisper_models/` directory:

```bash
python download_whipser.py
```

### 4. Prepare your data

Place your audio and label files in separate directories:

```
data/
├── audios/       # .wav files (16 kHz recommended)
└── labels/       # .json files (one per audio, with onset/offset/cluster arrays)
```

Each label JSON should have the following structure:

```json
{
  "onset": [0.5, 1.2, ...],
  "offset": [0.8, 1.5, ...],
  "cluster": ["m", "h", ...]
}
```

## Development

### Install with dev dependencies

```bash
pip install -e ".[dev]"
```

This installs everything from `pip install -e .` **plus** testing and linting tools (`pytest`, `ruff`).

### Run tests

```bash
pytest tests/ -v
```

### Run linting

```bash
ruff check .          # find issues
ruff format --check . # check formatting
```

CI runs these checks automatically on every push to `main` and on pull requests via GitHub Actions (see `.github/workflows/ci.yml`).

## WhisperSeg

Sequence-to-sequence approach from Gu et al. (https://github.com/nianlonggu/WhisperSeg) based on [Whisper](https://github.com/openai/whisper). The Whisper decoder generates text tokens encoding timestamps and cluster labels for each vocalization segment.

**Key classes:** `WhisperSegmenter`, `WhisperSegmenterFast` (CTranslate2-accelerated)

### Training

```bash
python -m lemurcalls.whisperseg.train \
    --initial_model_path <pretrained_model> \
    --model_folder <output_dir> \
    --audio_folder <audio_dir> \
    --label_folder <label_dir> \
    --num_classes <n> \
    --batch_size 4 \
    --learning_rate 3e-6 \
    --max_num_epochs 3
```

### Inference

```bash
python -m lemurcalls.whisperseg.infer \
    -d <data_dir> \
    -m <model_path> \
    -o <output_dir> \
    -n <num_trials>
```

## WhisperFormer

Dense prediction approach combining a frozen Whisper encoder with a lightweight decoder and classification/regression heads (inspired by ActionFormer). Predicts onset, offset, and call type for each time step. 

**Key class:** `WhisperFormer`

### Training

```bash
python -m lemurcalls.whisperformer.train \
    --checkpoint_path <pretrained_checkpoint> \
    --model_folder <output_dir> \
    --audio_folder <audio_dir> \
    --label_folder <label_dir> \
    --num_classes <n> \
    --whisper_size base \
    --batch_size 16 \
    --learning_rate 2e-4 \
    --max_num_epochs 40
```

### Inference

```bash
python -m lemurcalls.whisperformer.infer \
    --checkpoint_path <checkpoint> \
    --audio_folder <audio_dir> \
    --label_folder <label_dir> \
    --output_dir <output_dir> \
    --num_classes <n> \
    --whisper_size base
```
```bash
python -m lemurcalls.whisperformer.infer --checkpoint_path /projects/extern/CIDAS/cidas_digitalisierung_lehre/mthesis_sophie_dierks/dir.project/lemurcalls/lemurcalls/model_folder_new/final_model_20251205_030535/best_model.pth --audio_folder /mnt/lustre-grete/usr/u17327/final/audios_test --label_folder /mnt/lustre-grete/usr/u17327/final/jsons_test --output_dir /projects/extern/CIDAS/cidas_digitalisierung_lehre/mthesis_sophie_dierks/dir.project/lemurcalls/lemurcalls/model_folder_new/final_model_20251205_030535/sc --batch_size 4 --iou_threshold 0.4
```

### Visualize predictions

Generate per-segment plots showing the mel spectrogram, per-class model scores, ground truth labels (with quality annotations), and optionally extra labels (e.g. WhisperSeg predictions) for comparison.

```bash
python -m lemurcalls.visualize_predictions \
    --checkpoint_path <checkpoint> \
    --audio_folder <audio_dir> \
    --label_folder <gt_label_dir> \
    --pred_label_folder <whisperformer_pred_dir> \
    --output_dir <output_dir> \
    --threshold 0.35
```

To include a second set of predictions (e.g. from WhisperSeg) as an additional row:

```bash
python -m lemurcalls.visualize_predictions \
    --checkpoint_path <checkpoint> \
    --audio_folder <audio_dir> \
    --label_folder <gt_label_dir> \
    --pred_label_folder <whisperformer_pred_dir> \
    --extra_label_folder <whisperseg_pred_dir> \
    --output_dir <output_dir>
```

| Argument | Required | Description |
|---|---|---|
| `--checkpoint_path` | yes | Path to the trained WhisperFormer `.pth` checkpoint |
| `--audio_folder` | yes | Directory with `.wav` files |
| `--label_folder` | yes | Directory with ground truth `.json` labels |
| `--pred_label_folder` | yes | Directory with WhisperFormer prediction `.json` files |
| `--extra_label_folder` | no | Directory with additional prediction `.json` files (e.g. WhisperSeg) |
| `--output_dir` | no | Output directory for plots (default: `inference_outputs`) |
| `--threshold` | no | Score threshold (default: 0.35) |
| `--whisper_size` | no | `base` or `large` (auto-detected from checkpoint if omitted) |

The first 3 segments of each audio file are plotted and saved as `.png` files in a timestamped subdirectory.

### Filter results by SNR and maximale amplitude (recommended)
```bash
python -m lemurcalls.whisperformer.postprocessing.filter_labels_by_snr\
    --audio_folder <audio_dir>
    --label_folder <label_dir> \
    --output_dir <output_dir> \
    --snr_threshold <-1> \
    --amplitude_threshold <0.035>
```

## Utilities

The `lemurcalls.util` subpackage provides tools for:

- **Data preparation:** `make_json.py`, `split_wavs.py`, `trim_wavs.py`, `balance_cuts.py`
- **Annotation processing:** `process_pre_annotated.py`, `process_post_annotated.py`, `make_raven_table.py`
- **Evaluation:** `confusion_framewise.py`, `summarise_eval_res.py`, `all_cm.py`
- **Statistics:** `anno_stats.py`, `log_stat_parse.py`
