from __future__ import annotations

import argparse
import json
import math
import os
import time
from contextlib import nullcontext
from pathlib import Path

import numpy as np
import torch

from cs336_basics.data import get_batch, load_checkpoint, save_checkpoint
from cs336_basics.nn import AdamW, TransformerLM, cross_entropy, gradient_clipping, lr_cosine_schedule


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a decoder-only Transformer language model.")
    parser.add_argument("--train-data", required=True, help="Path to a 1D .npy array of token IDs.")
    parser.add_argument("--val-data", required=True, help="Path to a 1D .npy array of token IDs.")
    parser.add_argument("--out-dir", default="runs/debug", help="Directory for logs and checkpoints.")
    parser.add_argument("--resume", default=None, help="Checkpoint path to resume from.")

    parser.add_argument("--vocab-size", type=int, required=True)
    parser.add_argument("--context-length", type=int, default=256)
    parser.add_argument("--d-model", type=int, default=512)
    parser.add_argument("--d-ff", type=int, default=1344)
    parser.add_argument("--num-layers", type=int, default=4)
    parser.add_argument("--num-heads", type=int, default=16)
    parser.add_argument("--rope-theta", type=float, default=10000.0)

    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--steps", type=int, default=10000)
    parser.add_argument("--eval-iters", type=int, default=100)
    parser.add_argument("--eval-interval", type=int, default=500)
    parser.add_argument("--log-interval", type=int, default=10)
    parser.add_argument("--checkpoint-interval", type=int, default=1000)

    parser.add_argument("--max-lr", type=float, default=3e-4)
    parser.add_argument("--min-lr", type=float, default=3e-5)
    parser.add_argument("--warmup-iters", type=int, default=1000)
    parser.add_argument("--weight-decay", type=float, default=0.1)
    parser.add_argument("--beta1", type=float, default=0.9)
    parser.add_argument("--beta2", type=float, default=0.95)
    parser.add_argument("--eps", type=float, default=1e-8)
    parser.add_argument("--max-grad-norm", type=float, default=1.0)

    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--dtype", choices=["float32", "bfloat16"], default="bfloat16")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num-threads", type=int, default=None)
    return parser.parse_args()


def load_tokens(path: str | os.PathLike) -> np.ndarray:
    return np.load(path, mmap_mode="r")


def write_jsonl(path: Path, record: dict) -> None:
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=True) + "\n")


def make_amp_context(use_amp: bool):
    return torch.autocast(device_type="cuda", dtype=torch.bfloat16) if use_amp else nullcontext()


def move_optimizer_state_to_device(optimizer: torch.optim.Optimizer, device: str) -> None:
    for state in optimizer.state.values():
        for key, value in state.items():
            if torch.is_tensor(value):
                state[key] = value.to(device)


@torch.no_grad()
def estimate_loss(
    model: TransformerLM,
    dataset: np.ndarray,
    batch_size: int,
    context_length: int,
    device: str,
    eval_iters: int,
    use_amp: bool,
) -> float:
    model.eval()
    losses = []
    for _ in range(eval_iters):
        x, y = get_batch(dataset, batch_size, context_length, device)
        with make_amp_context(use_amp):
            logits = model(x)
            loss = cross_entropy(logits.reshape(-1, logits.shape[-1]), y.reshape(-1))
        losses.append(loss.item())
    model.train()
    return float(sum(losses) / len(losses))


def main() -> None:
    args = parse_args()
    if args.num_threads is not None:
        torch.set_num_threads(args.num_threads)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    log_path = out_dir / "train_log.jsonl"
    config_path = out_dir / "config.json"
    config_path.write_text(json.dumps(vars(args), indent=2, sort_keys=True), encoding="utf-8")

    train_data = load_tokens(args.train_data)
    val_data = load_tokens(args.val_data)
    if len(train_data) <= args.context_length or len(val_data) <= args.context_length:
        raise ValueError("train and val datasets must be longer than context_length")

    model = TransformerLM(
        vocab_size=args.vocab_size,
        context_length=args.context_length,
        d_model=args.d_model,
        num_layers=args.num_layers,
        num_heads=args.num_heads,
        d_ff=args.d_ff,
        rope_theta=args.rope_theta,
    ).to(args.device)
    optimizer = AdamW(
        model.parameters(),
        lr=args.max_lr,
        betas=(args.beta1, args.beta2),
        eps=args.eps,
        weight_decay=args.weight_decay,
    )

    start_step = 0
    if args.resume is not None:
        start_step = load_checkpoint(args.resume, model, optimizer)
        move_optimizer_state_to_device(optimizer, args.device)

    use_amp = args.device.startswith("cuda") and args.dtype == "bfloat16"
    start_time = time.time()
    model.train()

    for step in range(start_step, args.steps):
        lr = lr_cosine_schedule(step, args.max_lr, args.min_lr, args.warmup_iters, args.steps)
        for group in optimizer.param_groups:
            group["lr"] = lr

        x, y = get_batch(train_data, args.batch_size, args.context_length, args.device)
        optimizer.zero_grad(set_to_none=True)
        with make_amp_context(use_amp):
            logits = model(x)
            loss = cross_entropy(logits.reshape(-1, logits.shape[-1]), y.reshape(-1))
        loss.backward()
        gradient_clipping(model.parameters(), args.max_grad_norm)
        optimizer.step()

        iteration = step + 1
        if iteration % args.log_interval == 0 or iteration == 1:
            elapsed = time.time() - start_time
            record = {
                "step": iteration,
                "split": "train",
                "loss": loss.item(),
                "ppl": math.exp(min(loss.item(), 20.0)),
                "lr": lr,
                "elapsed_sec": elapsed,
                "tokens": iteration * args.batch_size * args.context_length,
            }
            write_jsonl(log_path, record)
            print(json.dumps(record, ensure_ascii=True), flush=True)

        if iteration % args.eval_interval == 0 or iteration == args.steps:
            val_loss = estimate_loss(
                model,
                val_data,
                args.batch_size,
                args.context_length,
                args.device,
                args.eval_iters,
                use_amp,
            )
            record = {
                "step": iteration,
                "split": "val",
                "loss": val_loss,
                "ppl": math.exp(min(val_loss, 20.0)),
                "lr": lr,
                "elapsed_sec": time.time() - start_time,
                "tokens": iteration * args.batch_size * args.context_length,
            }
            write_jsonl(log_path, record)
            print(json.dumps(record, ensure_ascii=True), flush=True)

        if iteration % args.checkpoint_interval == 0 or iteration == args.steps:
            save_checkpoint(model, optimizer, iteration, out_dir / f"ckpt_{iteration:06d}.pt")
            save_checkpoint(model, optimizer, iteration, out_dir / "ckpt_latest.pt")


if __name__ == "__main__":
    main()
