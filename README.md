# omni-flash-qwen

This repo contains a tiny, runnable scaffold to:

1. Run **TTS inference** from a locally cached Hugging Face model directory.
2. Fine-tune the **TTS component** (parameter-efficient) on your custom text–audio pairs using **LoRA**.

> Model path you mentioned:
> `/home/fudo/.cache/huggingface/hub/models--Qwen--Qwen3-Omni-30B-A3B-Instruct/`

## What you’ll get

- `scripts/tts_infer.py`: loads the model from a local path and attempts to synthesize speech.
- `scripts/prepare_dataset.py`: turns a folder of `text + wav` pairs into an HF dataset.
- `scripts/tts_lora_train.py`: LoRA training scaffold (Accelerate + PEFT) with audio preprocessing.

## Setup

Create a venv and install deps:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -U pip
pip install -r requirements.txt
```

## 1) TTS inference (smoke test)

```bash
python scripts/tts_infer.py \
  --model_path "/home/fudo/.cache/huggingface/hub/models--Qwen--Qwen3-Omni-30B-A3B-Instruct/" \
  --text "Hello! This is a TTS smoke test." \
  --out_wav out.wav
```

Notes:
- Qwen3-Omni is model-specific; depending on your installed `transformers` version and how the model is packaged, the audio generation API may be exposed as a custom `AutoModel` class or may require additional components.
- If the script prints that it can’t find the audio generation method, it will also print the discovered model methods to help you wire the right call.

## 2) Prepare a text–audio dataset

Expected input structure (example):

```
my_data/
  clips/
    0001.wav
    0002.wav
  metadata.csv
```

Where `metadata.csv` has:

```csv
file_name,text
0001.wav,Hello there.
0002.wav,Another sample.
```

Prepare dataset:

```bash
python scripts/prepare_dataset.py \
  --data_dir my_data \
  --clips_dir clips \
  --metadata_csv metadata.csv \
  --out_dir data/tts_dataset
```

## 3) LoRA fine-tune (scaffold)

```bash
accelerate config
accelerate launch scripts/tts_lora_train.py \
  --model_path "/home/fudo/.cache/huggingface/hub/models--Qwen--Qwen3-Omni-30B-A3B-Instruct/" \
  --dataset_dir data/tts_dataset \
  --output_dir outputs/tts_lora \
  --num_train_epochs 1 \
  --learning_rate 1e-4
```

### Important

Because Qwen3-Omni’s TTS stack can be packaged differently across versions, `tts_lora_train.py` is written to:

- Load the model and **discover likely TTS submodules**.
- Apply LoRA to a conservative set of common projection layers.
- Fail fast with actionable diagnostics if the expected audio/token heads aren’t present.

If you paste the console output of `tts_infer.py` (the “introspection” section), I can lock the scripts to the exact correct model API for your installed versions.

