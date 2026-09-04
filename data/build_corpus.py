"""Build a small public redistributable teaching corpus (Polybolos-authored)."""
from __future__ import annotations

from pathlib import Path

OUT = Path(__file__).resolve().parent / "corpus.txt"

PARAS = [
    "Polybolos Institute studies how defense systems fail under stress: decision latency, saturation, denied communications, and human-on-the-loop authority when the network is gone.",
    "Command Core runs a six-engine decision chain: Threat, Rules of Engagement, Economics, Ethics, Authority, and Course of Action. Kinetics and advisory accessories sit outside the Core chain.",
    "Human-on-the-Loop means the commander remains identifiable, bounded, observable, interruptible, and accountable. A human at the start of a process is not the same as human command through the fight.",
    "Standing Orders preserve engagement authority when communications are degraded. Authority must not evaporate mid-cycle because the link is gone.",
    "Domain language models can be trained on workstation-class GPUs. Path A builds from first principles for full auditability. Path B adapts open-weight models locally for stronger domain fluency.",
    "Operational security for sensitive work requires local training and local inference. Queries that leave the building are visible to a vendor.",
    "The Transformer uses token embeddings, positional signals, multi-head self-attention, residual connections, layer normalization, and a feed-forward network with GELU activation.",
    "Scaled dot-product attention computes softmax of Q times K transpose over square root of d_k, then multiplies by V. Multi-head attention attends in several subspaces at once.",
    "AdamW with decoupled weight decay is a standard optimizer for language model training. Mixed precision reduces activation memory on modern GPUs.",
    "Byte-pair encoding and related tokenizers map text to integer indices. For teaching models, byte-level vocabularies keep the pipeline simple and portable.",
    "Instruction tuning teaches a base model to treat input as a command. High-quality instruction pairs matter more than noisy volume.",
    "Constitutional alignment and preference methods can steer model behavior. Human-on-the-loop authority stays outside the model. Language models advise. Commanders decide.",
    "Denied-comms C2 requires edge-native fusion and deterministic decision paths. Assured command continues when commercial cloud dependency is unacceptable.",
    "Evaluation records model size, corpus description, training steps, hardware, validation loss, and a fixed domain instruction-check set.",
    "This corpus is authored by Polybolos Institute for public teaching use under the MIT license of the accompanying repository. It contains no controlled unclassified information and no operational logs.",
]

# Repeat and permute lightly to give the nano model enough tokens to learn patterns.
BLOCKS = []
for i in range(40):
    for p in PARAS:
        BLOCKS.append(p)
        BLOCKS.append(f"Teaching note {i}: {p}")
    BLOCKS.append(
        "Path A nano models learn local statistical structure from curated domain text. "
        "They are not frontier commercial systems. They prove ownership of weights on a desk."
    )

text = "\n\n".join(BLOCKS) + "\n"
OUT.write_text(text, encoding="utf-8")
print(f"Wrote {OUT} chars={len(text)} bytes={OUT.stat().st_size}")
