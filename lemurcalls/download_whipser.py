from transformers import WhisperModel, WhisperFeatureExtractor
import os

os.makedirs("whisper_models", exist_ok=True)

s = "base, large"
name = f"openai/whisper-{s}"

WhisperModel.from_pretrained(name).save_pretrained(
    f"whisper_models/whisper_{s}"
)
WhisperFeatureExtractor.from_pretrained(name).save_pretrained(
    f"whisper_models/whisper_{s}"
)

print(f"Saved whisper_{s}")
