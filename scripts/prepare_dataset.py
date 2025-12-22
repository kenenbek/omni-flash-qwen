#!/usr/bin/env python
"""Prepare a text–audio dataset for TTS fine-tuning.

Input:
- A directory with a `metadata.csv` mapping wav filenames to text.
- A subfolder containing the wav files.

Output:
- An on-disk Hugging Face `datasets` dataset with columns:
  - "text": str
  - "audio": datasets.Audio

This stays intentionally generic. The training script will do the final
resampling/normalization/tokenization.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from datasets import Audio, Dataset


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--data_dir", type=str, required=True)
    p.add_argument("--clips_dir", type=str, default="clips")
    p.add_argument("--metadata_csv", type=str, default="metadata.csv")
    p.add_argument("--out_dir", type=str, required=True)
    p.add_argument("--sample_rate", type=int, default=16000, help="Target sampling rate stored in the dataset")
    args = p.parse_args()

    data_dir = Path(args.data_dir)
    clips_dir = data_dir / args.clips_dir
    meta_path = data_dir / args.metadata_csv
    out_dir = Path(args.out_dir)
    out_dir.parent.mkdir(parents=True, exist_ok=True)

    if not clips_dir.exists():
        raise FileNotFoundError(clips_dir)
    if not meta_path.exists():
        raise FileNotFoundError(meta_path)

    # Load CSV without pandas to keep deps light.
    import csv

    rows = []
    with meta_path.open("r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        if "file_name" not in reader.fieldnames or "text" not in reader.fieldnames:
            raise ValueError("metadata.csv must contain headers: file_name,text")
        for r in reader:
            wav_path = clips_dir / r["file_name"]
            if not wav_path.exists():
                raise FileNotFoundError(wav_path)
            rows.append({"text": r["text"], "audio": str(wav_path)})

    ds = Dataset.from_list(rows)
    ds = ds.cast_column("audio", Audio(sampling_rate=args.sample_rate))

    out_dir.mkdir(parents=True, exist_ok=True)
    ds.save_to_disk(str(out_dir))
    print(f"Saved dataset to: {out_dir} (rows={len(ds)})")


if __name__ == "__main__":
    main()

