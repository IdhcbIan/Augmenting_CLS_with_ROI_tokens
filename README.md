# CLS + Buddy Tokens (minimal reference implementation)

### Overview
This micro-repository demonstrates how to:
1. Load the **DINO-v2 ViT-B/14 “register” checkpoint** from `timm`.
2. Extract the five *cue* tokens (*[CLS]* + 4 *register* tokens).
3. Generate one **Buddy token** per cue via 3 × 3 mean pooling around the nearest image patch.
4. Train the 10-token representation with a **ColBERT-style max-similarity contrastive loss**.

The code is intentionally compact (~250 LOC) so you can drop individual parts into other projects without extra glue code.

### Prerequisites
* Python ≥ 3.9  
* CUDA-capable GPU (optional but recommended)  
* Packages listed in `requirements.txt`

Create and activate a virtual environment, then install:

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

### Repo Structure

| File                 | Purpose                                                                                                                             |
| -------------------- | ----------------------------------------------------------------------------------------------------------------------------------- |
| **model\_utils.py**  | Loads DINO-v2 ViT-B/14-register, applies preprocessing, returns *(cue\_tokens, patch\_tokens)*.                                     |
| **buddy\_pool.py**   | `BuddyPool` module: finds each cue’s nearest patch, mean-pools its 3 × 3 neighborhood to form a Buddy token.                        |
| **maxsim\_loss.py**  | `MaxSimLoss`: hinge-margin version (α = 0.1) of the ColBERT max-sim loss.                                                           |
| **train.py**         | Demo driver: builds random (anchor, positive, negative) triplets, runs cue → Buddy pipeline, optimises `BuddyPool` for a few steps. |
| **requirements.txt** | Minimal dependency list (PyTorch ≥ 2.2, `timm`, etc.).                                                                              |
| **README.md**        | This document.                                                                                                                      |
## End to End Flow

image ─▶ preprocessing ─▶ ViT-Reg backbone ─▶
      └──────── cue tokens (5) ──┐
                                 ├─▶ BuddyPool ──▶ ROI tokens (5)
patch grid (16×16) ─────────────┘
cue + Buddy tokens (10) ────────▶ MaxSimLoss
