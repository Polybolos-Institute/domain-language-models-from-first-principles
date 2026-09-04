#!/usr/bin/env python3
"""Train a Path A GPT (nano or small) on the bundled public corpus."""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch

from polybolos_lm.model import CONFIGS, GPT, count_parameters


def encode_bytes(text: str) -> torch.Tensor:
    return torch.tensor(list(text.encode("utf-8")), dtype=torch.long)


def main() -> None:
    ap = argparse.ArgumentParser(description="Train Polybolos Path A domain LM")
    ap.add_argument("--config", choices=sorted(CONFIGS), default="nano")
    ap.add_argument("--data", type=Path, default=Path("data/corpus.txt"))
    ap.add_argument("--out", type=Path, default=Path("checkpoints"))
    ap.add_argument("--steps", type=int, default=2000)
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--lr", type=float, default=3e-3)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--eval-every", type=int, default=200)
    args = ap.parse_args()

    if not args.data.exists():
        raise SystemExit(f"Missing corpus: {args.data}. Run: python data/build_corpus.py")

    text = args.data.read_text(encoding="utf-8")
    data = encode_bytes(text)
    n = int(len(data) * 0.9)
    train_data, val_data = data[:n], data[n:]

    cfg = CONFIGS[args.config]
    device = torch.device(args.device)
    model = GPT(cfg).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.1)
    params = count_parameters(model)
    print(f"config={args.config} params={params:,} device={device} tokens={len(data):,}")

    def get_batch(split: str):
        src = train_data if split == "train" else val_data
        ix = torch.randint(0, len(src) - cfg.block_size - 1, (args.batch_size,))
        x = torch.stack([src[i : i + cfg.block_size] for i in ix])
        y = torch.stack([src[i + 1 : i + cfg.block_size + 1] for i in ix])
        return x.to(device), y.to(device)

    @torch.no_grad()
    def estimate_loss():
        model.eval()
        out = {}
        for split in ("train", "val"):
            losses = []
            for _ in range(20):
                xb, yb = get_batch(split)
                _, loss = model(xb, yb)
                losses.append(loss.item())
            out[split] = sum(losses) / len(losses)
        model.train()
        return out

    args.out.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    model.train()
    for step in range(1, args.steps + 1):
        xb, yb = get_batch("train")
        _, loss = model(xb, yb)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        if step == 1 or step % args.eval_every == 0 or step == args.steps:
            losses = estimate_loss()
            print(
                f"step {step:5d}/{args.steps} loss={loss.item():.4f} "
                f"train={losses['train']:.4f} val={losses['val']:.4f}"
            )

    elapsed = time.time() - t0
    ckpt_path = args.out / f"polybolos_{args.config}.pt"
    meta = {
        "config_name": args.config,
        "config": cfg.__dict__,
        "params": params,
        "steps": args.steps,
        "device": str(device),
        "elapsed_sec": round(elapsed, 1),
        "final_val_loss": estimate_loss()["val"],
        "corpus": str(args.data).replace("\\", "/"),
        "tokenizer": "utf-8-bytes",
        "paper": "PI-WP-2026-040410",
    }
    torch.save(
        {"model": model.state_dict(), "config": cfg.__dict__, "meta": meta},
        ckpt_path,
    )
    meta_path = args.out / f"polybolos_{args.config}_meta.json"
    meta_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print(f"Wrote {ckpt_path} ({ckpt_path.stat().st_size} bytes)")
    print(f"Wrote {meta_path}")


if __name__ == "__main__":
    main()
