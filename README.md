# Augmenting CLS with ROI Tokens for Image Retrieval

<div align="center">

**🚀 A scalable implementation of multi-vector image retrieval using [CLS] + Region of Interest (ROI) tokens**

*Powered by DINOv2 ViT-B/14 with Register Tokens and ColBERT-style similarity scoring*

</div>

## 🎯 Overview

This repository implements a novel approach to image retrieval by augmenting traditional [CLS] tokens with Region of Interest (ROI) tokens. The method leverages DINOv2's register tokens as "cues" to identify important image regions through **buddy pooling**, creating a 10-token multi-vector representation optimized with ColBERT-style contrastive learning.

### Key Innovation: Buddy Pooling
- **5 Cue Tokens**: [CLS] + 4 DINOv2 register tokens
- **5 ROI Tokens**: Generated by finding the most similar patch for each cue and pooling its 3×3 neighborhood
- **10-Token Representation**: Combined cues + ROIs for rich image representation
- **ColBERT Scoring**: Max-similarity matching for flexible multi-vector retrieval

## 🔧 Technical Implementation

### Core Components

| Module | Purpose | Key Features |
|--------|---------|--------------|
| **`TrainableMultiVectorEncoder`** | Main model architecture | DINOv2 backbone + trainable projection layer |
| **`BuddyPool`** | ROI token generation | 3×3 mean pooling around nearest patches |
| **`TripletColbertLoss`** | Training objective | ColBERT-style max-similarity with margin |
| **Modal Apps** | Cloud GPU training | Multi-GPU support on A100s |


### Buddy Pooling Algorithm
```python
def buddy_pool(cue_token, patch_grid):
    # Find most similar patch for each cue
    similarities = cue_token @ patch_grid.flatten(2, 3).T
    best_patch_idx = similarities.argmax()
    
    # Extract 3x3 neighborhood around best patch
    h, w = best_patch_idx // grid_width, best_patch_idx % grid_width
    roi_region = patch_grid[max(0, h-1):min(H, h+2), 
                           max(0, w-1):min(W, w+2)]
    
    # Mean pool to create ROI token
    return roi_region.mean(dim=(0, 1))
```

## 📊 Supported Datasets

| Dataset | Classes | Images | Description | Status |
|---------|---------|--------|-------------|---------|
| **CUB-200-2011** | 200 | 11,788 | Fine-grained bird species | ✅ Full support |
| **Flowers** | 17 | 1,360 | Flower categories | ✅ Full support |
| **Oxford Buildings** | 17 | 5,063 | Landmark retrieval | ✅ Full support |

## 🚀 Modal Cloud Training

This project leverages [Modal](https://modal.com/) for scalable cloud GPU training with automatic environment management.

### Quick Start
```bash
# Install Modal CLI
pip install modal

# Setup Modal account
modal setup

# Run multi-GPU training on CUB-200-2011
modal run Modal_CUB200/modal_app_cub_multi.py

# Run on Oxford Flowers
modal run Modal_Flowers/modal_app_Flowers.py

# Run on Oxford Buildings
modal run Modal_ROxford/modal_app_roxford_duo.py
```

### Push DataBase to Modal
```bash
modal volume create [Volume_Name]

# Run multi-GPU training on CUB-200-2011
modal run Modal_CUB200/modal_app_cub_multi.py

# Run on Oxford Flowers
modal run Modal_Flowers/modal_app_Flowers.py

# Run on Oxford Buildings
modal run Modal_ROxford/modal_app_roxford_duo.py
```

### Available Modal Applications

#### CUB-200-2011 Training Options
- **`modal_app_cub_multi.py`** - Multi-GPU training ((N)×A100-80GB) ⭐ **Recommended**

#### Multi-Dataset Support
- **`Modal_Flowers/modal_app_Flowers.py`** - Oxford Flowers training
- **`Modal_ROxford/modal_app_roxford_duo.py`** - Oxford Buildings with 4×A100s

### Training Configuration

```python
# Example: CUB Multi-GPU Training
@app.function(
    gpu="A100-80GB:2",     # 2×A100-80GB GPUs
    timeout=3600           # 1 hour timeout
)
def main(
    steps=20,             # Training steps
    batch_size=512,        # Per-GPU batch size (×2 = 256 total)
    lr=1e-5,              # Learning rate
    eval_batch_size=100     # Evaluation batch size
):
```

## 📈 Performance & Results

### Key Metrics
- **Recall@1**: Primary retrieval accuracy metric
- **Recall@2**: Top-2 retrieval performance
- **Recall@4**: Top-4 retrieval performance
- **Recall@8**: Top-8 retrieval performance (Not Aways Used)
- **ColBERT Score**: Multi-token similarity measure

## 🔬 Technical Details

### Model Architecture
- **Backbone**: DINOv2 ViT-B/14 with register tokens (768-dim)
- **Input Size**: 224×224 (configurable to 518×518)

### Training Dynamics
- **Loss Function**: Triplet loss with ColBERT scoring
- **Optimizer**: AdamW with weight decay 0.01
- **Learning Rate**: 1e-5 to 3e-4 (dataset dependent)
- **Batch Sizes**: 128-512 depending on GPU configuration and DataSet

## 📁 Repository Structure

```
Augmenting_CLS_with_ROI_tokens/
├── README.md                    # This file
├── Modal_Call.txt              # Quick Modal usage examples
│
├── Modal_CUB200/               # CUB-200-2011 implementations
│   ├── modal_app_cub_multi.py  # Multi-GPU training (recommended)
│   ├── modal_app_cub_duo.py    # Dual-GPU variant
│   ├── modal_app_cub_trainable.py
│   ├── model_utils.py          # Model definitions
│   ├── buddy_pool.py           # ROI pooling implementation
│   ├── maxsim_loss.py          # ColBERT loss functions
│   └── train_cub.py            # Local training script
│
├── Modal_Flowers/              # Oxford Flowers implementations
│   ├── modal_app_Flowers.py    # Flowers training on Modal
│   └── [shared modules]        # buddy_pool.py, maxsim_loss.py, etc.
│
├── Modal_ROxford/              # Oxford Buildings implementations
│   ├── modal_app_roxford_duo.py # Multi-GPU buildings training
│   ├── modal_app_roxford.py    # Single-GPU variant
│   └── [shared modules]
│
├── DataSets/                   # Dataset storage and converters
│   ├── convert_flowers.py      # Flowers format converter
│   ├── convert_roxford.py      # Oxford Buildings converter
│   ├── CUB_200_2011/          # Original CUB dataset
│   ├── Flowers_converted/      # Processed flowers
│   └── roxford5k_converted/    # Processed buildings
│
└── References/                 # Original research code
    └── Original_Code/          # Reference implementations
```

<div align="center">

**⭐ Star this repository if you find it useful!**

</div>
