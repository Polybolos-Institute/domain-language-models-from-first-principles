# Aleta AI

**Product:** Aleta AI (Path A teaching domain language model)  
**Document ID:** PI-WP-2026-040410  
**Revision:** A (September 2026)  
**Classification:** Unclassified // Approved for Public Release  
**Author:** Mark Brown, Polybolos Institute

Aleta AI is the public downloadable Path A package from Polybolos Institute: train a GPT-style domain language model from scratch, or download the pretrained nano checkpoint and generate offline.

## Quick start (download and run)

```bash
pip install -r requirements.txt
python generate.py --ckpt checkpoints/polybolos_nano.pt --prompt "Path A builds"
```

If `checkpoints/` is empty, pull the latest GitHub Release asset, or train locally (below).

## Train from scratch

```bash
python data/build_corpus.py
python train.py --config nano --steps 2000
python generate.py --ckpt checkpoints/polybolos_nano.pt
```

| Config | Approx params | Default use |
|--------|---------------|-------------|
| `nano` | ~0.8M | Fast CPU/GPU teaching run |
| `small` | ~25M+ | Workstation GPU recommended |

Hardware: CPU works for nano. A workstation GPU shortens wall time. Record steps and val loss in the Release notes / `EVAL.md`.

## Layout

| Path | Role |
|------|------|
| `polybolos_lm/model.py` | GPT decoder (byte vocab 256) |
| `train.py` | Train loop |
| `generate.py` | Inference CLI |
| `data/build_corpus.py` | Builds public Polybolos-authored corpus |
| `data/corpus.txt` | Generated teaching text (MIT redistributable) |
| `checkpoints/` | `polybolos_nano.pt` (+ meta JSON) |

## Paper

- PDF: [Polybolos_Institute_WP-2026-040410_Building_Artificial_Intelligence.pdf](Polybolos_Institute_WP-2026-040410_Building_Artificial_Intelligence.pdf)
- Site: https://polybolos.org/PI-WP-2026-040410.html

Path A = from-scratch ownership (this package). Path B (7B LoRA) is described in the paper and is not Aleta AI v1.

## Bounds

- Teaching / research domain LM. Not Command HOTL. Not engagement authority.
- Does not claim to beat commercial frontier systems.
- Corpus is unclassified public teaching text only.

## Inquiries

Contact@Polybolos.org

Polybolos Institute - 501(c)(3) - Dallas, Texas  
https://polybolos.org
