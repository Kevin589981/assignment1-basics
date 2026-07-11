from __future__ import annotations

import argparse
import json
import pickle
import time
from pathlib import Path

import numpy as np

from cs336_basics.tokenizer import Tokenizer


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Encode UTF-8 text into a 1D NumPy token ID array.")
    parser.add_argument("--input", required=True, help="Path to UTF-8 text.")
    parser.add_argument("--tokenizer", required=True, help="Tokenizer pickle from scripts/train_tokenizer.py.")
    parser.add_argument("--output", required=True, help="Output .npy path.")
    parser.add_argument("--dtype", choices=["uint16", "uint32", "int64"], default="uint16")
    parser.add_argument("--metadata", default=None, help="Optional JSON metadata output path.")
    return parser.parse_args()


def load_tokenizer(path: str | Path) -> Tokenizer:
    with open(path, "rb") as f:
        payload = pickle.load(f)
    return Tokenizer(payload["vocab"], payload["merges"], payload.get("special_tokens"))


def main() -> None:
    args = parse_args()
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    tokenizer = load_tokenizer(args.tokenizer)

    start = time.time()
    with open(args.input, encoding="utf-8") as f:
        token_ids = list(tokenizer.encode_iterable(f))
    elapsed = time.time() - start

    max_id = max(token_ids) if token_ids else 0
    dtype = np.dtype(args.dtype)
    if max_id > np.iinfo(dtype).max:
        raise ValueError(f"max token id {max_id} does not fit in dtype {dtype}")
    array = np.asarray(token_ids, dtype=dtype)
    np.save(output, array)

    input_bytes = Path(args.input).stat().st_size
    metadata = {
        "input": str(args.input),
        "tokenizer": str(args.tokenizer),
        "output": str(output),
        "dtype": str(dtype),
        "num_tokens": int(array.size),
        "max_token_id": int(max_id),
        "input_bytes": int(input_bytes),
        "bytes_per_token": float(input_bytes / array.size) if array.size else None,
        "tokens_per_sec": float(array.size / elapsed) if elapsed > 0 else None,
        "elapsed_sec": elapsed,
    }
    if args.metadata is not None:
        metadata_path = Path(args.metadata)
        metadata_path.parent.mkdir(parents=True, exist_ok=True)
        metadata_path.write_text(json.dumps(metadata, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(metadata, ensure_ascii=True), flush=True)


if __name__ == "__main__":
    main()
