# Augmenting CLS with ROI Tokens for Image Retrieval

This repository contains research code for multi-vector visual retrieval. It augments the global `[CLS]` descriptor of a DINOv2 Vision Transformer with region-of-interest (ROI) descriptors selected from the patch-token grid.

The resulting token set is trained and compared with late-interaction (MaxSim) scoring. Experimental pipelines are provided for CUB-200-2011, Stanford Cars196, In-Shop Clothes Retrieval, and INSTRE.

This is research code intended to accompany experimental work, rather than a general-purpose retrieval library.

## Method

### Representation

The encoder uses `vit_base_patch14_reg4_dinov2.lvd142m` from `timm`, a DINOv2 ViT-B/14 backbone with one `[CLS]` token and four register tokens.

$$
C = \{c_0, \ldots, c_4\},
$$

Here, $c_0$ is `[CLS]`; the other four tokens are register tokens. Let $P \in \mathbb{R}^{H \times W \times d}$ denote the spatial patch-token grid.

For every cue $c_i$, the method selects the most similar patch token:

$$
(h_i, w_i) = \underset{(h,w)}{\arg\max}\; c_i^\top P_{h,w},
$$

It then average-pools a square neighborhood centered at $(h_i, w_i)$, producing one ROI token $r_i$ per cue. The final representation is the L2-normalized ten-token set:

$$
E(x) = [c_0, \ldots, c_4, r_0, \ldots, r_4].
$$

The trainable CUB, Cars196, and In-Shop implementations project 768-dimensional backbone tokens to 384 dimensions before pooling. They use 224-pixel inputs and border-aware pooling.

Their pooling radius is five patch locations, implemented by `roi_side = 10`. An interior neighborhood therefore contains up to 11 by 11 patch tokens. The frozen-feature utility in `Modal_CUB200/model_utils.py` is a separate 518-pixel, 3 by 3 pooling prototype.

### Retrieval objective

For query tokens $Q$ and candidate tokens $D$, the late-interaction score is:

$$
s(Q,D) = \sum_i \max_j q_i^\top d_j.
$$

The training pipelines use a margin triplet objective:

$$
\mathcal{L} = \max\left(0, m + s(Q,N) - s(Q,P)\right),
$$

Each batch contains multiple examples from the same class or item, enabling online hard-negative mining.

Buddy pooling makes a discrete patch selection. The implementation uses a straight-through estimator: the forward pass uses `argmax`, while the backward pass substitutes a low-temperature softmax surrogate.

## Repository layout

```text
.
├── Modal_CUB200/       CUB-200-2011 training and evaluation experiments
├── Modal_Cars196/      Stanford Cars196 training and evaluation experiments
├── Modal_In-Shop/      In-Shop Clothes Retrieval experiments
├── Modal_Instre/       INSTRE experiments
├── DataSets/           Optional local dataset mount point; not distributed here
└── References/          Preserved reference implementation
```

The most complete trainable entry points are:

| Dataset | Training script | Modal volume expected by the script |
|---|---|---|
| CUB-200-2011 | `Modal_CUB200/modal_app_cub_new.py` | `cub-data` |
| Stanford Cars196 | `Modal_Cars196/modal_app_cars_save.py` | `stanford_cars` |
| In-Shop Clothes Retrieval | `Modal_In-Shop/modal_app_InShop_train.py` | `In-Shop` |
| INSTRE | `Modal_Instre/modal_app_instre_save_hardnegative_estimator_strict.py` | `instre_converted` |

Several additional scripts record exploratory variants. For each experiment, report the exact script, commit, checkpoint, preprocessing, and hyperparameters.

## Installation

The Modal scripts construct their own container image. They install PyTorch, torchvision, timm 0.9.12, einops 0.7.0, Pillow, NumPy, and tqdm.

For local inspection or development, create a Python environment and install compatible versions:

```bash
python -m venv .venv
source .venv/bin/activate
pip install torch torchvision timm==0.9.12 einops==0.7.0 pillow numpy tqdm modal
```

The DINOv2 weights are requested through `timm` when the model is instantiated. Ensure that the environment can obtain the pretrained checkpoint, or configure the local model cache in advance.

## Data preparation

Datasets are not included in this repository. Download each dataset from its official source and comply with its license and terms of use.

The scripts expect data in a Modal volume mounted at `/mnt/data`.

For CUB-200-2011, the training script expects the standard annotation files and image tree:

```text
/mnt/data/CUB_200_2011/
├── classes.txt
├── images.txt
├── image_class_labels.txt
├── train_test_split.txt
└── images/
```

Create and populate the volume before launching a run. For example, after installing and authenticating the Modal CLI:

```bash
modal volume create cub-data
```

Use Modal's volume upload workflow to copy the prepared dataset into the volume. Adapt the volume name and on-volume directory to the selected dataset script.

## Running an experiment

Authenticate the Modal CLI once:

```bash
modal setup
```

Then launch a dataset-specific experiment from its directory so that the script's local source-file references resolve correctly:

```bash
cd Modal_CUB200
modal run modal_app_cub_new.py
```

The CUB script defaults are:

- 1,000 optimization steps
- 12 classes and 6 samples per class per batch
- hard-negative ratio of `0.1`
- triplet margin of `0.3`
- straight-through temperature of `0.05`
- AdamW with learning rate `1e-5` and weight decay `0.01`

Checkpoints and run metadata are written beneath `/mnt/data/Checkpoints/` in the mounted volume.

Before a paper-scale run, inspect the selected script's `main` function and record every overridden argument. GPU allocation, data paths, batch construction, and evaluation behavior differ across scripts.

## Evaluation and reporting

The repository includes utilities that report recall at selected ranks. The gallery/query protocol is script-dependent; for example, some utilities use training images as the gallery and held-out images as queries.

For each reported result, document:

- Dataset version, split, and any filtering of classes or images.
- Backbone identifier, pretrained-weight source, image resolution, projection dimension, and number of tokens.
- ROI neighborhood size and straight-through temperature.
- Optimizer, learning rate, margin, hard-negative ratio, batch composition, number of steps, random seeds, and hardware.
- Retrieval metric definition, query/gallery construction, and whether self-matches are excluded.
- Checkpoint selection criterion and the mean and variation over independent runs.

The code does not define one canonical benchmark protocol or a consolidated results table.

Do not treat values printed by an exploratory script as publication-ready results without validating the protocol and repeating the experiment.

## References

If this repository contributes to published work, add the paper citation here before release. The implementation relies on the following methodological foundations:

- M. Oquab et al. *DINOv2: Learning Robust Visual Features without Supervision*. 2023.
- O. Khattab and M. Zaharia. *ColBERT: Efficient and Effective Passage Search via Contextualized Late Interaction over BERT*. 2020.

## License and dataset terms

No project license file is currently included. Add an explicit license before public distribution. Dataset access and redistribution remain governed by the respective dataset providers.
