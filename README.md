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
│   ├── datautils.py         # Audio/label loading and slicing
│   ├── convert_hf_to_ct2.py # HuggingFace to CTranslate2 conversion
│   └── ...
├── tests/                   # Unit tests
├── pyproject.toml           # Package configuration & dependencies
└── README.md
```

## Installation

```bash
# Clone the repository
git clone <repo-url>
cd lemurcalls

# Install in editable mode
pip install -e .
```

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

## Utilities

The `lemurcalls.util` subpackage provides tools for:

- **Data preparation:** `make_json.py`, `split_wavs.py`, `trim_wavs.py`, `balance_cuts.py`
- **Annotation processing:** `process_pre_annotated.py`, `process_post_annotated.py`, `make_raven_table.py`
- **Evaluation:** `confusion_framewise.py`, `summarise_eval_res.py`, `all_cm.py`
- **Statistics:** `anno_stats.py`, `log_stat_parse.py`
