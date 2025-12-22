#!/usr/bin/env python
"""LoRA fine-tuning scaffold for the TTS component of a Qwen3-Omni model.

Important:
- Qwen3-Omni’s exact audio/TTS training interface can differ by packaging/version.
- This script focuses on the *engineering plumbing*: dataset loading, audio preprocessing,
  LoRA injection, and an Accelerate training loop.
- You will likely need to customize `_build_supervised_batch()` to match the model’s
  expected inputs/labels for TTS (e.g., codec tokens).

It tries to fail fast with clear diagnostics.
"""

from __future__ import annotations

import argparse
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import torch
from accelerate import Accelerator
from datasets import load_from_disk
from peft import LoraConfig, TaskType, get_peft_model
from torch.utils.data import DataLoader
from tqdm.auto import tqdm


def _pick_torch_dtype(dtype: str) -> torch.dtype:
    if dtype == "bf16":
        return torch.bfloat16
    if dtype == "fp16":
        return torch.float16
    return torch.float32


def _find_lora_targets(model: torch.nn.Module) -> List[str]:
    """Heuristic target modules; you should narrow this to the actual TTS stack."""
    # Common projection layer names.
    candidates = {
        "q_proj",
        "k_proj",
        "v_proj",
        "o_proj",
        "gate_proj",
        "up_proj",
        "down_proj",
        "Wqkv",
        "out_proj",
    }
    found = set()
    for name, module in model.named_modules():
        if module.__class__.__name__ in {"Linear", "Conv1d"}:
            leaf = name.split(".")[-1]
            if leaf in candidates:
                found.add(leaf)
    # If nothing matches, fall back to any Linear leaf names (last resort).
    if not found:
        for name, module in model.named_modules():
            if module.__class__.__name__ == "Linear":
                found.add(name.split(".")[-1])
        found = set(list(found)[:8])
    return sorted(found)


@dataclass
class Batch:
    input_ids: torch.Tensor
    attention_mask: torch.Tensor
    labels: torch.Tensor


def _normalize_audio(wav: np.ndarray) -> np.ndarray:
    wav = wav.astype(np.float32)
    if wav.ndim == 2:
        wav = wav.mean(axis=0)
    peak = float(np.max(np.abs(wav)) + 1e-8)
    wav = wav / peak
    return wav


def _build_supervised_batch(
    processor: Any,
    batch_examples: List[Dict[str, Any]],
    device: torch.device,
) -> Batch:
    """Turn raw dataset items into a training batch.

    Currently: a placeholder that trains a *text-only* language modeling objective
    on the prompt text, because the true TTS target representation is model-specific.

    To fine-tune *TTS*, you typically need to:
    - Convert audio waveform -> codec tokens (or model audio labels)
    - Construct a multimodal prompt that conditions on text and predicts audio tokens

    Replace this function once `tts_infer.py` introspection identifies the correct API.
    """

    texts = [ex["text"] for ex in batch_examples]

    if processor is None or not hasattr(processor, "__call__"):
        raise RuntimeError("Processor is required; AutoProcessor failed to load for this model")

    enc = processor(text=texts, return_tensors="pt", padding=True, truncation=True)
    input_ids = enc["input_ids"].to(device)
    attention_mask = enc.get("attention_mask", torch.ones_like(input_ids)).to(device)

    labels = input_ids.clone()
    return Batch(input_ids=input_ids, attention_mask=attention_mask, labels=labels)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_path", type=str, required=True)
    ap.add_argument("--dataset_dir", type=str, required=True)
    ap.add_argument("--output_dir", type=str, required=True)

    ap.add_argument("--per_device_train_batch_size", type=int, default=1)
    ap.add_argument("--gradient_accumulation_steps", type=int, default=8)
    ap.add_argument("--num_train_epochs", type=int, default=1)
    ap.add_argument("--learning_rate", type=float, default=1e-4)
    ap.add_argument("--max_steps", type=int, default=-1)
    ap.add_argument("--warmup_steps", type=int, default=50)
    ap.add_argument("--logging_steps", type=int, default=10)

    ap.add_argument("--lora_r", type=int, default=16)
    ap.add_argument("--lora_alpha", type=int, default=32)
    ap.add_argument("--lora_dropout", type=float, default=0.05)

    ap.add_argument("--dtype", type=str, default="bf16", choices=["bf16", "fp16", "fp32"])
    ap.add_argument("--seed", type=int, default=42)

    args = ap.parse_args()

    accelerator = Accelerator(gradient_accumulation_steps=args.gradient_accumulation_steps)
    torch.manual_seed(args.seed)

    model_path = args.model_path
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    accelerator.print(f"Loading model from: {model_path}")

    from transformers import AutoModelForCausalLM, AutoProcessor

    processor = AutoProcessor.from_pretrained(model_path, local_files_only=True, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        local_files_only=True,
        trust_remote_code=True,
        torch_dtype=_pick_torch_dtype(args.dtype),
        device_map=None,
    )

    # Enable grad checkpointing if supported.
    if hasattr(model, "gradient_checkpointing_enable"):
        model.gradient_checkpointing_enable()

    # Apply LoRA.
    targets = _find_lora_targets(model)
    accelerator.print("LoRA target_module candidates:", targets)

    lora_cfg = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        target_modules=targets,
        bias="none",
    )
    model = get_peft_model(model, lora_cfg)
    model.print_trainable_parameters()

    ds = load_from_disk(args.dataset_dir)

    def collate_fn(examples: List[Dict[str, Any]]) -> Dict[str, Any]:
        b = _build_supervised_batch(processor, examples, device=accelerator.device)
        return {"input_ids": b.input_ids, "attention_mask": b.attention_mask, "labels": b.labels}

    dl = DataLoader(
        ds,
        batch_size=args.per_device_train_batch_size,
        shuffle=True,
        collate_fn=collate_fn,
    )

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate)

    model, optimizer, dl = accelerator.prepare(model, optimizer, dl)

    # Steps schedule.
    steps_per_epoch = math.ceil(len(dl) / args.gradient_accumulation_steps)
    total_steps = steps_per_epoch * args.num_train_epochs
    if args.max_steps > 0:
        total_steps = min(total_steps, args.max_steps)

    def lr_lambda(step: int) -> float:
        if step < args.warmup_steps:
            return float(step) / float(max(1, args.warmup_steps))
        return 1.0

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_lambda)

    model.train()
    global_step = 0

    pbar = tqdm(total=total_steps, disable=not accelerator.is_local_main_process)

    for epoch in range(args.num_train_epochs):
        for step, batch in enumerate(dl):
            with accelerator.accumulate(model):
                out = model(**batch)
                loss = out.loss
                accelerator.backward(loss)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad(set_to_none=True)

            if accelerator.sync_gradients:
                global_step += 1
                pbar.update(1)

                if global_step % args.logging_steps == 0:
                    accelerator.print({"step": global_step, "loss": float(loss.detach().cpu())})

                if args.max_steps > 0 and global_step >= args.max_steps:
                    break

        if args.max_steps > 0 and global_step >= args.max_steps:
            break

    accelerator.wait_for_everyone()

    # Save adapters.
    if accelerator.is_main_process:
        accelerator.print("Saving LoRA adapter to:", str(out_dir))
        unwrapped = accelerator.unwrap_model(model)
        unwrapped.save_pretrained(str(out_dir), safe_serialization=True)
        # Save processor for convenience.
        processor.save_pretrained(str(out_dir))

    accelerator.print("Done.")


if __name__ == "__main__":
    main()

