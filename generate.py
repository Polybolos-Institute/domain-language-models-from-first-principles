#!/usr/bin/env python3
"""Generate text from a trained Path A checkpoint."""
from __future__ import annotations

import argparse
from pathlib import Path

import torch

from polybolos_lm.model import GPT, GPTConfig


def main() -> None:
    ap = argparse.ArgumentParser(description="Generate from Polybolos Path A checkpoint")
    ap.add_argument(
        "--ckpt",
        type=Path,
        default=Path("checkpoints/polybolos_nano.pt"),
    )
    ap.add_argument("--prompt", default="Path A builds")
    ap.add_argument("--tokens", type=int, default=200)
    ap.add_argument("--temperature", type=float, default=0.9)
    ap.add_argument("--top-k", type=int, default=40)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

    if not args.ckpt.exists():
        raise SystemExit(f"Missing checkpoint: {args.ckpt}")

    blob = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    cfg = GPTConfig(**blob["config"])
    model = GPT(cfg)
    model.load_state_dict(blob["model"])
    device = torch.device(args.device)
    model.to(device)
    model.eval()

    prompt_bytes = list(args.prompt.encode("utf-8"))
    idx = torch.tensor([prompt_bytes], dtype=torch.long, device=device)
    out = model.generate(
        idx,
        max_new_tokens=args.tokens,
        temperature=args.temperature,
        top_k=args.top_k,
    )[0].tolist()
    text = bytes(out).decode("utf-8", errors="replace")
    print(text)


if __name__ == "__main__":
    main()
