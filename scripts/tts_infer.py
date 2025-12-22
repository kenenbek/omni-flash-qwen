#!/usr/bin/env python
"""TTS smoke test for a locally cached Qwen3-Omni model.

This script is intentionally defensive:
- It loads from a local directory (no network).
- It tries a few common audio-generation entrypoints.
- If it can’t find the right method (model packaging differs by version),
  it prints useful introspection so you can wire the correct call.

Usage:
  python scripts/tts_infer.py --model_path /path/to/model --text "..." --out_wav out.wav
"""

from __future__ import annotations

import argparse
import inspect
import json
import os
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import numpy as np
import torch


def _pick_device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def _safe_list_methods(obj: Any, prefix: str = "") -> Dict[str, str]:
    methods: Dict[str, str] = {}
    for name in dir(obj):
        if prefix and not name.startswith(prefix):
            continue
        if name.startswith("_"):
            continue
        try:
            attr = getattr(obj, name)
        except Exception:
            continue
        if callable(attr):
            methods[name] = type(attr).__name__
    return dict(sorted(methods.items()))


def _write_wav(path: str, wav: np.ndarray, sample_rate: int) -> None:
    import soundfile as sf

    wav = np.asarray(wav, dtype=np.float32)
    if wav.ndim == 2 and wav.shape[0] == 1:
        wav = wav[0]
    sf.write(path, wav, sample_rate)


@torch.inference_mode()
def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--text", type=str, required=True)
    parser.add_argument("--out_wav", type=str, default="out.wav")
    parser.add_argument("--dtype", type=str, default="auto", choices=["auto", "bf16", "fp16", "fp32"])
    parser.add_argument("--max_new_tokens", type=int, default=1024)
    args = parser.parse_args()

    model_path = args.model_path
    if not os.path.exists(model_path):
        raise FileNotFoundError(model_path)

    device = _pick_device()

    if args.dtype == "auto":
        torch_dtype = torch.bfloat16 if (device == "cuda" and torch.cuda.is_bf16_supported()) else torch.float16
        if device == "cpu":
            torch_dtype = torch.float32
    elif args.dtype == "bf16":
        torch_dtype = torch.bfloat16
    elif args.dtype == "fp16":
        torch_dtype = torch.float16
    else:
        torch_dtype = torch.float32

    print(f"Loading from: {model_path}")
    print(f"Device: {device} | dtype: {torch_dtype}")

    # Local-only loading.
    from transformers import AutoConfig, AutoModelForCausalLM, AutoProcessor

    config = AutoConfig.from_pretrained(model_path, local_files_only=True, trust_remote_code=True)
    processor = None
    try:
        processor = AutoProcessor.from_pretrained(model_path, local_files_only=True, trust_remote_code=True)
    except Exception as e:
        print("AutoProcessor load failed (this may be OK):", repr(e))

    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        local_files_only=True,
        trust_remote_code=True,
        torch_dtype=torch_dtype,
        device_map="auto" if device != "cpu" else None,
    )
    model.eval()

    print("\n=== Model/Config summary ===")
    print("model class:", model.__class__.__name__)
    print("config class:", config.__class__.__name__)
    print("config model_type:", getattr(config, "model_type", None))

    # Attempt 1: a dedicated TTS / audio generation method.
    candidate_methods = [
        "generate_audio",
        "tts",
        "text_to_speech",
        "infer_tts",
        "synthesize",
    ]

    for method_name in candidate_methods:
        if hasattr(model, method_name) and callable(getattr(model, method_name)):
            fn = getattr(model, method_name)
            print(f"\nFound candidate method: model.{method_name}{inspect.signature(fn)}")
            try:
                out = fn(args.text)
                wav, sr = _extract_wav_and_sr(out)
                _write_wav(args.out_wav, wav, sr)
                print(f"Wrote: {args.out_wav} (sr={sr}, samples={len(wav)})")
                return
            except Exception as e:
                print(f"Calling model.{method_name} failed:", repr(e))

    # Attempt 2: a pipeline if available.
    try:
        from transformers import pipeline

        # Some models expose TTS via a task name like "text-to-speech".
        tts_pipe = pipeline(
            task="text-to-speech",
            model=model,
            tokenizer=getattr(processor, "tokenizer", None),
            feature_extractor=getattr(processor, "feature_extractor", None),
            device=0 if device == "cuda" else -1,
        )
        out = tts_pipe(args.text)
        wav, sr = _extract_wav_and_sr(out)
        _write_wav(args.out_wav, wav, sr)
        print(f"Wrote: {args.out_wav} (sr={sr}, samples={len(wav)})")
        return
    except Exception as e:
        print("\nPipeline(text-to-speech) attempt failed:", repr(e))

    # Attempt 3: generic generate() with a multimodal processor.
    if processor is not None:
        print("\nTrying generic generate() path (may not produce audio on this model packaging)…")
        try:
            inputs = processor(text=args.text, return_tensors="pt")
            inputs = {k: v.to(model.device) for k, v in inputs.items()}
            gen = model.generate(**inputs, max_new_tokens=args.max_new_tokens)
            if hasattr(processor, "decode"):
                decoded = processor.decode(gen[0], skip_special_tokens=True)
                print("Decoded text output:")
                print(decoded)
            else:
                print("Generated token ids:", gen[0][:50].tolist(), "…")
        except Exception as e:
            print("generate() attempt failed:", repr(e))

    print("\n=== Introspection (share this output if you want me to lock in the exact API) ===")
    print("Public callable methods on model (subset):")
    methods = _safe_list_methods(model)
    # Keep output bounded.
    for k in list(methods.keys())[:200]:
        print(" -", k)

    print("\nTop-level model submodules (subset):")
    for name, _mod in list(model.named_children())[:80]:
        print(" -", name)

    print("\nProcessor fields:")
    if processor is None:
        print(" - <None>")
    else:
        for k in ["tokenizer", "feature_extractor", "image_processor", "audio_processor"]:
            print(f" - {k}:", hasattr(processor, k))

    raise SystemExit(
        "Could not find a working TTS entrypoint automatically. "
        "This model’s TTS API varies by packaging/version. "
        "Paste the introspection output and I’ll wire the correct call."
    )


def _extract_wav_and_sr(out: Any) -> Tuple[np.ndarray, int]:
    """Try to normalize common TTS outputs to (waveform, sample_rate)."""

    if isinstance(out, tuple) and len(out) == 2:
        wav, sr = out
        return np.asarray(wav), int(sr)

    if isinstance(out, dict):
        # Common HF pipeline output: {"audio": array, "sampling_rate": 16000}
        if "audio" in out and ("sampling_rate" in out or "sample_rate" in out):
            sr = out.get("sampling_rate", out.get("sample_rate"))
            return np.asarray(out["audio"]), int(sr)

        # Alternate keys.
        for ak in ["wav", "waveform"]:
            if ak in out:
                sr = out.get("sampling_rate", out.get("sample_rate", 16000))
                return np.asarray(out[ak]), int(sr)

    # Some pipelines return a numpy array directly.
    if isinstance(out, np.ndarray):
        return out, 16000

    # Last resort: JSON-serializable debug.
    try:
        print("Raw TTS output type:", type(out))
        print("Raw TTS output (repr):", repr(out)[:1000])
        if hasattr(out, "__dict__"):
            print("Raw TTS output __dict__:", json.dumps(out.__dict__, default=str)[:2000])
    except Exception:
        pass

    raise TypeError("Don’t know how to extract waveform + sample rate from the model output")


if __name__ == "__main__":
    main()

