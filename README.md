# Domain Language Models From First Principles

**Document ID:** PI-WP-2026-040410  
**Revision:** A (September 2026)  
**Classification:** Unclassified // Approved for Public Release  
**Author:** Mark Brown, Polybolos Institute

Public white paper: how defense research organizations build and own domain AI on workstation hardware.

## Paper

- PDF: [Polybolos_Institute_WP-2026-040410_Building_Artificial_Intelligence.pdf](Polybolos_Institute_WP-2026-040410_Building_Artificial_Intelligence.pdf)
- Site abstract: https://polybolos.org/PI-WP-2026-040410.html
- Publications: https://polybolos.org/publications.html

## What it covers

Defense research organizations can own domain artificial intelligence today. The ingredients are mathematical literacy, Python proficiency, a curated public domain corpus, and a workstation-class GPU.

Two paths:

| Path | What you get |
|------|----------------|
| **A - From scratch** | GPT-style domain language model (~3M / ~50M / ~350M). Maximum auditability. Complete ownership from corpus to checkpoint. |
| **B - Open weight + local adaptation** | Continue-pretrain or LoRA / QLoRA an open-weight base inside an air-gapped environment. Strong domain performance with local inference. |

Both paths keep queries on hardware you control. Teaching stack follows the public Transformer / nanoGPT lineage (~500 lines of Python).

## Why this repo exists

The full paper is already on polybolos.org. This repository makes the PDF easy to clone, cite, and share under MIT, next to Polybolos Institute open samples.

## Bounds

- Research / construction guide for **domain** language models on workstation GPUs.
- Not Command HOTL source. Not engagement authorization.
- Not a claim that a 50-350M model is a frontier commercial LLM.
- Public recipe uses public / unclassified / licensed corpora only.

## Inquiries

Contact@Polybolos.org

Polybolos Institute - 501(c)(3) - Dallas, Texas  
https://polybolos.org
