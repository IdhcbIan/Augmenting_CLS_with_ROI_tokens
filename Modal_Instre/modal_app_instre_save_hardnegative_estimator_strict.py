import modal
import random
import time
from pathlib import Path
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import trange, tqdm
import timm
from torchvision import transforms
from PIL import Image
from einops import rearrange
import os
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import Dataset, DataLoader
import numpy as np
from datetime import datetime


# Build Modal Image including local Python source code
image = (
    modal.Image.debian_slim()
    .pip_install("torch", "torchvision", "tqdm", "timm==0.9.12", "einops==0.7.0", "pillow", "numpy")
    .add_local_file("buddy_pool.py", "/root/buddy_pool.py")
    .add_local_file("maxsim_loss.py", "/root/maxsim_loss.py")
    .add_local_file("modal_app_instre_train.py", "/root/modal_app_instre_train.py")
)

# Define Modal App with dataset volume
app = modal.App(
    "INSTRE FULL HARD NEGATIVE TRAIN ESTIMATOR",
    image=image,
    volumes={"/mnt/data": modal.Volume.from_name("instre_converted")}
)


class StraightThroughArgmax(torch.autograd.Function):
    """
    Straight-through estimator for argmax:
    - Forward pass: uses argmax (discrete selection)
    - Backward pass: uses softmax with low temperature for differentiability
    """
    @staticmethod
    def forward(ctx, similarities, temperature=0.1):
        # Forward: discrete argmax selection
        idx = similarities.argmax(dim=-1)
        
        # Save for backward pass
        ctx.save_for_backward(similarities)
        ctx.temperature = temperature
        
        return idx
    
    @staticmethod
    def backward(ctx, grad_output):
        # Backward: use softmax with low temperature
        similarities, = ctx.saved_tensors
        temperature = ctx.temperature
        
        # Compute softmax probabilities with low temperature
        softmax_probs = F.softmax(similarities / temperature, dim=-1)
        
        # Gradient flows through softmax probabilities
        grad_similarities = grad_output.unsqueeze(-1) * softmax_probs
        
        return grad_similarities, None  # None for temperature (no gradient needed)


class TrainableMultiVectorEncoder(nn.Module):
    """TRAINABLE Multi-vector encoder with straight-through estimator for differentiable buddy pooling."""
    
    def __init__(self):
        super().__init__()
        MODEL_NAME = "vit_base_patch14_reg4_dinov2.lvd142m"
        
        # Configuration matching original
        self.embed_dim = 768
        self.num_registers = 4
        self.img_size = model_img_size
        self.roi_side = 3
        
        # Create the model - trainable with correct image size
        self.backbone = timm.create_model(MODEL_NAME, pretrained=True, num_classes=0, img_size=self.img_size)
        
        # Add a small projection layer to make it clearly trainable
        self.projection = nn.Linear(self.embed_dim, self.embed_dim)
        
    def _buddy_pool(self, cue, patches2d):
        """Improved buddy pooling with straight-through estimator for differentiability."""
        B, H, W, d = patches2d.shape
        flat = rearrange(patches2d, "b h w d -> b (h w) d")
        sim = torch.matmul(cue.unsqueeze(1), flat.transpose(1, 2)).squeeze(1)
        
        # Use straight-through estimator: argmax forward, softmax backward
        idx = StraightThroughArgmax.apply(sim, 0.05)  # temperature = 0.05 for sharper gradients
        
        h = idx // W
        w = idx % W
        r = self.roi_side // 2
        roi = []
        for b in range(B):
            hs = slice(max(0, h[b]-r), min(H, h[b]+r+1))
            ws = slice(max(0, w[b]-r), min(W, w[b]+r+1))
            roi.append(patches2d[b, hs, ws, :].mean(dim=(0, 1)))
        return torch.stack(roi)
        
    def forward(self, x):
        """Clean forward pass with differentiable buddy pooling."""
        # Forward through backbone
        tokens = self.backbone.forward_features(x)
        
        # Apply projection (trainable layer)
        tokens = self.projection(tokens)
        
        # Extract tokens following original structure
        cls_tok = tokens[:, 0:1, :]  # CLS token: (B, 1, D)
        regs_tok = tokens[:, 1:1 + self.num_registers, :]  # Register tokens: (B, 4, D)
        patch_tok = tokens[:, 1 + self.num_registers:, :]  # Patch tokens: (B, N, D)
        
        # Reshape patch tokens to spatial grid
        g = int(self.img_size // 14)  # ViT-B/14 grid size
        patches2d = rearrange(patch_tok, "b (h w) d -> b h w d", h=g, w=g)
        
        # Combine CLS and register tokens to form cues
        cues = torch.cat([cls_tok, regs_tok], dim=1)  # (B, 5, D)
        
        # Apply buddy pooling to get ROIs with straight-through estimator
        rois = torch.stack([self._buddy_pool(cues[:, i], patches2d)
                           for i in range(cues.size(1))], dim=1)
        
        # Combine cues and ROIs
        toks = torch.cat([cues, rois], dim=1)  # (B, 10, D)
        
        # Normalize
        return F.normalize(toks, dim=-1)


def colbert_score(X, Y):
    """ColBERT scoring function from the original implementation."""
    return torch.einsum("bnd,bmd->bnm", X, Y).max(dim=-1).values.sum(dim=-1)


class TripletColbertLoss(nn.Module):
    """Triplet loss using ColBERT scoring from the original implementation."""
    
    def __init__(self, margin=0.2):
        super().__init__()
        self.margin = margin
        
    def forward(self, anchor, positive, negative):
        pos_score = colbert_score(anchor, positive)
        neg_score = colbert_score(anchor, negative)
        loss = F.relu(neg_score - pos_score + self.margin)
        return loss.mean()


class HardNegativeTripletLoss(nn.Module):
    """Hard negative triplet loss with online hard negative mining."""
    
    def __init__(self, margin=0.2, hard_negative_ratio=0.2):
        super().__init__()
        self.margin = margin
        self.hard_negative_ratio = hard_negative_ratio  # Fraction of hardest negatives to use
        
    def forward(self, embeddings, labels):
        """
        Args:
            embeddings: (B, num_tokens, dim) - batch embeddings
            labels: (B,) - class labels for each sample
        """
        batch_size = embeddings.size(0)
        device = embeddings.device
        
        # Compute all pairwise similarities using ColBERT score
        similarities = torch.zeros(batch_size, batch_size, device=device)
        for i in range(batch_size):
            for j in range(batch_size):
                if i != j:
                    similarities[i, j] = colbert_score(
                        embeddings[i:i+1], embeddings[j:j+1]
                    ).item()
        
        losses = []
        
        for i in range(batch_size):
            anchor_label = labels[i]
            anchor_emb = embeddings[i:i+1]
            
            # Find positive samples (same class, excluding self)
            pos_mask = (labels == anchor_label) & (torch.arange(batch_size, device=device) != i)
            if not pos_mask.any():
                continue  # Skip if no positives available
                
            # Find negative samples (different class)
            neg_mask = labels != anchor_label
            if not neg_mask.any():
                continue  # Skip if no negatives available
            
            # Select hardest positive (lowest similarity among positives)
            pos_indices = torch.where(pos_mask)[0]
            pos_similarities = similarities[i, pos_indices]
            hardest_pos_idx = pos_indices[pos_similarities.argmin()]
            positive_emb = embeddings[hardest_pos_idx:hardest_pos_idx+1]
            
            # Select hard negatives (highest similarities among negatives)
            neg_indices = torch.where(neg_mask)[0]
            neg_similarities = similarities[i, neg_indices]
            
            # Take top hard_negative_ratio of negatives
            num_hard_negs = max(1, int(len(neg_indices) * self.hard_negative_ratio))
            hard_neg_indices = neg_indices[neg_similarities.argsort(descending=True)[:num_hard_negs]]
            
            # Compute loss for each hard negative
            for neg_idx in hard_neg_indices:
                negative_emb = embeddings[neg_idx:neg_idx+1]
                
                pos_score = colbert_score(anchor_emb, positive_emb)
                neg_score = colbert_score(anchor_emb, negative_emb)
                loss = F.relu(neg_score - pos_score + self.margin)
                losses.append(loss)
        
        if not losses:
            return torch.tensor(0.0, device=device, requires_grad=True)
        
        return torch.stack(losses).mean()


class INSTREHardNegativeDataset(Dataset):
    """Dataset for INSTRE hard negative mining with balanced class sampling."""
    
    def __init__(self, train_paths, batch_size=32, classes_per_batch=8, samples_per_class=4):
        self.batch_size = batch_size
        self.classes_per_batch = classes_per_batch
        self.samples_per_class = samples_per_class
        
        # train_paths is already a dict of {class_name: [paths]}
        self.class_to_paths = train_paths
        self.classes = list(self.class_to_paths.keys())
        
        # Filter classes with enough samples
        min_samples = max(2, samples_per_class)
        self.classes = [cls for cls in self.classes if len(self.class_to_paths[cls]) >= min_samples]
        
        print(f" Hard Negative Dataset initialized:")
        print(f"   Total classes: {len(self.classes)}")
        print(f"   Classes per batch: {classes_per_batch}")
        print(f"   Samples per class: {samples_per_class}")
        print(f"   Effective batch size: {classes_per_batch * samples_per_class}")
        
        # Pre-compute dataset size (arbitrary large number for continuous sampling)
        self.dataset_size = 10000
        
    def __len__(self):
        return self.dataset_size
    
    def __getitem__(self, idx):
        """Sample a balanced batch for hard negative mining."""
        # Select random classes for this batch
        selected_classes = random.sample(self.classes, min(self.classes_per_batch, len(self.classes)))
        
        batch_images = []
        batch_labels = []
        
        for class_idx, class_id in enumerate(selected_classes):
            # Sample images from this class
            class_paths = self.class_to_paths[class_id]
            sampled_paths = random.sample(class_paths, min(self.samples_per_class, len(class_paths)))
            
            for path in sampled_paths:
                image = _load_image(path)
                if image is not None:
                    batch_images.append(image)
                    batch_labels.append(class_idx)  # Use index as label for easier processing
        
        # Pad or truncate to exact batch size if needed
        while len(batch_images) < self.batch_size:
            # Add random samples to fill batch
            random_class = random.choice(selected_classes)
            random_path = random.choice(self.class_to_paths[random_class])
            image = _load_image(random_path)
            if image is not None:
                batch_images.append(image)
                batch_labels.append(selected_classes.index(random_class))
        
        # Truncate if we have too many
        batch_images = batch_images[:self.batch_size]
        batch_labels = batch_labels[:self.batch_size]
        
        return torch.stack(batch_images), torch.tensor(batch_labels, dtype=torch.long)


#------// Model Params //-----------------
model_img_size = 224

def _load_image(path):
    """Load a PIL image and preprocess it to tensor."""
    preprocess = transforms.Compose([
        transforms.Resize(model_img_size),
        transforms.CenterCrop(model_img_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                           std=[0.229, 0.224, 0.225]),
    ])
    try:
        img = Image.open(path).convert("RGB")
        return preprocess(img)
    except (FileNotFoundError, OSError) as e:
        print(f"  Skipping file {path}: {e}")
        return None


def parse_instre(root: Path):
    """
    Parse INSTRE converted metadata and return training/test image path dictionaries
    """
    print(" Starting INSTRE parsing...")
    
    cls_map = {}
    for line in (root / "classes.txt").read_text().splitlines():
        cid, cname = line.split()
        cls_map[int(cid)] = cname

    img_to_cid = {}
    for line in (root / "image_class_labels.txt").read_text().splitlines():
        iid, cid = line.split()
        img_to_cid[int(iid)] = int(cid)

    img_map = {}
    skipped_files = 0
    for line in (root / "images.txt").read_text().splitlines():
        iid, rel = line.split()
        full_path = root / rel
        # Skip system files and check if file exists
        if not rel.startswith('._') and full_path.exists():
            img_map[int(iid)] = str(full_path)
        else:
            skipped_files += 1

    print(10 *"=")
    print(f"Img map length: {len(img_map)}")
    print(10 *"=")

    train_ids = set()
    for line in (root / "train_test_split.txt").read_text().splitlines():
        iid, flag = line.split()
        if int(flag):
            train_ids.add(int(iid))

    # Use class names as keys for better readability
    train_paths = {cls_map[int(c)]: [] for c in cls_map.keys()}
    test_paths  = {cls_map[int(c)]: [] for c in cls_map.keys()}
    
    # Create comprehensive directory-based class mapping
    path_to_class = {}  # Track full paths to class (directory-based)
    dir_to_class = {}   # Track directory to class mapping
    
    for iid, path in img_map.items():
        cid = img_to_cid[iid]
        class_name = cls_map[cid]
        full_path = str(path)
        
        # Extract directory from path for class assignment
        path_obj = Path(path)
        directory = path_obj.parent.name  # Get the directory name
        
        # Add to path mapping (most reliable)
        path_to_class[full_path] = class_name
        
        # Track directory to class mapping
        dir_to_class[directory] = class_name
        
        # Add to train/test paths
        (train_paths[class_name] if iid in train_ids else test_paths[class_name]).append(path)

    # Filter classes that have sufficient training and test samples
    train_paths = {c: ps for c, ps in train_paths.items() if len(ps) >= 2}
    test_paths  = {c: ps for c, ps in test_paths.items() if c in train_paths and len(ps) >= 1}
    
    # Filter filename mapping to only include valid classes
    valid_classes = set(train_paths.keys())
    filename_to_class = {f: c for f, c in path_to_class.items() if c in valid_classes}
    
    # DEBUG: Print dataset statistics
    print(f" Dataset Statistics:")
    print(f"   - Total classes: {len(train_paths)}")
    train_sizes = [len(paths) for paths in train_paths.values()]
    print(f"   - Train images per class: min={min(train_sizes)}, max={max(train_sizes)}, avg={sum(train_sizes)/len(train_sizes):.1f}")
    print(f"   - Total train images: {sum(train_sizes)}")
    test_sizes = [len(paths) for paths in test_paths.values()]
    print(f"   - Total test images: {sum(test_sizes)}")
    print(f"   - Filename-to-class mappings: {len(filename_to_class)}")

    # OPTIMIZATION: Convert all paths to strings to reduce Path object overhead
    print(" Converting paths to strings for faster access...")
    train_paths = {k: [str(p) for p in v] for k, v in train_paths.items()}
    test_paths = {k: [str(p) for p in v] for k, v in test_paths.items()}
    
    return train_paths, test_paths, filename_to_class, path_to_class, dir_to_class


@app.function(
    gpu="A100-80GB:2",  # 2 A100-80GB GPUs
    timeout=7500  # 2.5 hour timeout
)
def main(
    instre_root: str = "/mnt/data/instre_converted",  # Uses full INSTRE dataset
    steps: int = 1000,
    batch_size: int = 64,   # Smaller batch for hard negative mining
    report_interval: int = 1,
    eval_batch_size: int = 256,  
    lr: float = 1e-5,
    classes_per_batch: int = 12,  # MORE classes per batch for harder negatives
    samples_per_class: int = 6,   # Fewer samples per class for more diversity
    hard_negative_ratio: float = 0.02,  # ULTRA AGGRESSIVE: Use top 10% hardest negatives only
    margin: float = 0.5,  # LARGER margin for more challenging training
    temperature: float = 0.05  # Temperature for straight-through estimator
):
    """
    Train INSTRE model using hard negative mining on the FULL INSTRE dataset 
    with straight-through estimator for differentiable buddy pooling.
    
    Args:
        instre_root: Path to instre_converted directory
        steps: Number of training steps
        batch_size: Effective batch size (classes_per_batch * samples_per_class)
        report_interval: Steps between progress reports
        eval_batch_size: Batch size for evaluation
        lr: Learning rate
        classes_per_batch: Number of different classes in each batch
        samples_per_class: Number of samples per class in each batch
        hard_negative_ratio: Fraction of hardest negatives to use (0.0-1.0)
        margin: Triplet loss margin
        temperature: Temperature for straight-through estimator (lower = sharper gradients)
    """
    print(f" Starting AGGRESSIVE INSTRE Hard Negative Mining Training with Straight-Through Estimator")
    print(f"   Dataset: Full INSTRE (no threshold filtering)")
    print(f"   AGGRESSIVE Hard Negative Mining: {classes_per_batch} classes × {samples_per_class} samples = {classes_per_batch * samples_per_class} per batch")
    print(f"   Hard negative ratio: {hard_negative_ratio} (top {int(hard_negative_ratio*100)}% hardest - ULTRA AGGRESSIVE)")
    print(f"   Margin: {margin} (LARGE margin for challenging training)")
    print(f"   Straight-through estimator temperature: {temperature}")
    print("="*60)
    
    # Setup multi-GPU environment
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
    
    instre_root_path = Path(instre_root)
    
    # Load full INSTRE dataset (no threshold filtering)
    train_paths, test_paths, filename_to_class, path_to_class, dir_to_class = parse_instre(instre_root_path)
    
    if not train_paths:
        raise ValueError("No train files found in INSTRE dataset")
    
    print(f" Full INSTRE Dataset Loaded:")
    print(f"   Training classes: {len(train_paths)}")
    print(f"   Total train images: {sum(len(paths) for paths in train_paths.values())}")
    print(f"   Total test images: {sum(len(paths) for paths in test_paths.values())}")
    
    # Setup multi-GPU
    num_gpus = torch.cuda.device_count()
    print(f" Using {num_gpus} GPUs: {[torch.cuda.get_device_name(i) for i in range(num_gpus)]}")
    
    # Create model with straight-through estimator
    print(" Creating model with differentiable buddy pooling (straight-through estimator)")
    model = TrainableMultiVectorEncoder()
    
    if num_gpus > 1:
        device = torch.device("cuda:0")
        model = model.to(device)
        model = nn.DataParallel(model)
        print(f" Model wrapped with DataParallel across {num_gpus} GPUs")
        effective_batch_size = batch_size * num_gpus
    else:
        device = torch.device("cuda:0")
        model = model.to(device)
        effective_batch_size = batch_size
    
    print(f"Model has {sum(p.numel() for p in model.parameters() if p.requires_grad):,} trainable parameters")
    
    # Calculate actual effective batch size for hard negative mining
    actual_batch_size = classes_per_batch * samples_per_class
    if num_gpus > 1:
        effective_batch_size = actual_batch_size * num_gpus
    else:
        effective_batch_size = actual_batch_size
        
    print(f" AGGRESSIVE Hard Negative Mining Configuration:")
    print(f"   Classes per batch: {classes_per_batch} (MORE classes = harder negatives)")
    print(f"   Samples per class: {samples_per_class} (fewer per class = more diversity)")
    print(f"   Actual batch size: {actual_batch_size}")
    print(f"   Effective batch size: {effective_batch_size} (gpus: {num_gpus})")
    print(f"   Hard negative ratio: {hard_negative_ratio} (TOP {int(hard_negative_ratio*100)}% HARDEST - ULTRA SELECTIVE)")
    print(f"   Margin: {margin} (LARGE margin for challenging training)")
    print(f"   Learning rate: {lr} (constant - no scheduler)")
    print(f"   Straight-through temperature: {temperature}")
    
    # Setup optimizer and hard negative mining loss (constant LR)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    criterion = HardNegativeTripletLoss(margin=margin, hard_negative_ratio=hard_negative_ratio)
    
    # Setup dataset and dataloader for hard negative mining
    print(" Setting up Hard Negative Mining Dataset...")
    dataset = INSTREHardNegativeDataset(
        train_paths, 
        batch_size=actual_batch_size,
        classes_per_batch=classes_per_batch,
        samples_per_class=samples_per_class
    )
    
    # Dataloader with batch_size=1 since dataset returns full batches
    num_workers = min(4, actual_batch_size // 8)  # Fewer workers since we're doing more complex sampling
    dataloader = DataLoader(
        dataset, 
        batch_size=1,  # Dataset returns complete batches, so we use batch_size=1
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=True if num_workers > 0 else False,
    )
    
    print(f" Hard Negative Dataset ready:")
    print(f"   Total classes: {len(dataset.classes)}")
    print(f"   Dataset steps: {len(dataset)}")
    print(f"   Dataloader workers: {num_workers}")
    
    # Training loop with hard negative mining and straight-through estimator
    hist = []
    
    for i, batch in enumerate(tqdm(dataloader, desc="Hard Negative Mining Training (Straight-Through)", total=steps)):
        if i >= steps:
            break
            
        # Unpack batch (dataset returns (images, labels) as a tuple)
        images, labels = batch[0].squeeze(0), batch[1].squeeze(0)  # Remove dataloader batch dimension
        images, labels = images.to(device), labels.to(device)
        
        print(f" Step {i+1}/{steps}: Processing batch with {len(torch.unique(labels))} classes...")
        print(f"   Batch shape: {images.shape}, Labels: {labels.shape}")
        print(f"   Using differentiable buddy pooling (straight-through estimator)")

        optimizer.zero_grad()
        
        # Forward pass to get embeddings (now with differentiable buddy pooling)
        embeddings = model(images)  # (B, num_tokens, dim)
        
        # Compute hard negative mining loss
        loss = criterion(embeddings, labels)
        
        if loss.item() > 0:  # Only backprop if we have valid loss
            loss.backward()
            
            # Gradient clipping for stability
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            optimizer.step()
        
        hist.append(loss.item())
        
        print(f"Current Loss: {loss.item():.6f}")
        print(f"Current LR: {optimizer.param_groups[0]['lr']:.8f} (constant)")

        if (i + 1) % report_interval == 0:
            avg_loss = sum(hist[-report_interval:]) / report_interval
            print(f"[step {i+1:4d}] avg loss: {avg_loss:.4f} ( AGGRESSIVE hard_neg_ratio: {hard_negative_ratio}, margin: {margin})")

    print(" Training completed!")
    print(f"Final loss: {hist[-1]:.6f}")
    
    # Save model checkpoint
    print(" Saving model checkpoint...")
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    checkpoint_dir = Path(f"/mnt/data/Checkpoints/aggressive_hardneg_full_estimator_{timestamp}")
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    print(f" Created checkpoint directory: {checkpoint_dir}")
    
    # Prepare model for saving
    model_to_save = model.module if hasattr(model, 'module') else model
    
    # Save complete checkpoint
    checkpoint = {
        'model_state_dict': model_to_save.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'training_history': hist,
        'final_loss': hist[-1] if hist else 0.0,
        'training_config': {
            'steps': steps,
            'batch_size': batch_size,
            'actual_batch_size': actual_batch_size,
            'effective_batch_size': effective_batch_size,
            'classes_per_batch': classes_per_batch,
            'samples_per_class': samples_per_class,
            'hard_negative_ratio': hard_negative_ratio,
            'margin': margin,
            'lr': lr,
            'model_img_size': model_img_size,
            'num_gpus': num_gpus,
            'train_classes_count': len(train_paths),
            'total_train_images': sum(len(paths) for paths in train_paths.values()),
            'total_test_images': sum(len(paths) for paths in test_paths.values()),
            'dataset_classes': len(dataset.classes),
            'training_method': 'aggressive_hard_negative_mining_full_dataset_straight_through_estimator',
            'straight_through_temperature': temperature,
            'differentiable_buddy_pooling': True,
            'learning_rate_schedule': 'constant'
        },
        'timestamp': timestamp
    }
    
    checkpoint_path = checkpoint_dir / "checkpoint.pth"
    torch.save(checkpoint, checkpoint_path)
    print(f" Saved complete checkpoint to: {checkpoint_path}")
    
    # Save model weights
    model_weights_path = checkpoint_dir / "model_weights.pth"
    torch.save(model_to_save.state_dict(), model_weights_path)
    print(f" Saved model weights to: {model_weights_path}")
    
    # Save configuration
    config_path = checkpoint_dir / "config.txt"
    with open(config_path, 'w') as f:
        f.write(f"INSTRE AGGRESSIVE Hard Negative Mining Training - FULL DATASET - Straight-Through Estimator - {timestamp}\n")
        f.write("=" * 80 + "\n")
        f.write(f"Training Method: AGGRESSIVE Hard Negative Mining + Straight-Through Estimator\n")
        f.write(f"Dataset: Full INSTRE (no threshold filtering)\n")
        f.write(f"Steps: {steps}\n")
        f.write(f"Batch Configuration:\n")
        f.write(f"  - Classes per batch: {classes_per_batch}\n")
        f.write(f"  - Samples per class: {samples_per_class}\n")
        f.write(f"  - Actual batch size: {actual_batch_size}\n")
        f.write(f"  - Effective batch size: {effective_batch_size}\n")
        f.write(f" AGGRESSIVE Hard Negative Mining:\n")
        f.write(f"  - Hard negative ratio: {hard_negative_ratio} (top {int(hard_negative_ratio*100)}% - ULTRA SELECTIVE)\n")
        f.write(f"  - Triplet margin: {margin} (LARGE margin for challenging training)\n")
        f.write(f"Straight-Through Estimator:\n")
        f.write(f"  - Temperature: {temperature}\n")
        f.write(f"  - Differentiable buddy pooling: Enabled\n")
        f.write(f"  - Forward pass: Discrete argmax selection\n")
        f.write(f"  - Backward pass: Softmax with low temperature\n")
        f.write(f"Learning Rate: {lr} (constant - no scheduler)\n")
        f.write(f"Model Image Size: {model_img_size}\n")
        f.write(f"Number of GPUs: {num_gpus}\n")
        f.write(f"Final Loss: {hist[-1] if hist else 0.0:.6f}\n")
        f.write(f"Training Classes: {len(train_paths)}\n")
        f.write(f"Total Train Images: {sum(len(paths) for paths in train_paths.values())}\n")
        f.write(f"Total Test Images: {sum(len(paths) for paths in test_paths.values())}\n")
        f.write(f"Dataset Classes: {len(dataset.classes)}\n")
        f.write(f"Total Training Steps: {len(hist)}\n")
    print(f" Saved training config to: {config_path}")
    
    # Save dataset info for reference
    dataset_info_path = checkpoint_dir / "dataset_info.txt"
    with open(dataset_info_path, 'w') as f:
        f.write(f"Full INSTRE Dataset Information - Straight-Through Estimator\n")
        f.write("=" * 50 + "\n")
        f.write(f"Training Classes: {len(train_paths)}\n")
        f.write(f"Total Train Images: {sum(len(paths) for paths in train_paths.values())}\n")
        f.write(f"Total Test Images: {sum(len(paths) for paths in test_paths.values())}\n")
        f.write(f"\nStraight-Through Estimator Details:\n")
        f.write(f"Temperature: {temperature}\n")
        f.write(f"Forward Pass: Discrete argmax selection for buddy pooling\n")
        f.write(f"Backward Pass: Softmax with temperature {temperature} for gradients\n")
        f.write(f"Benefit: Enables gradient flow through buddy pooling operation\n")
        f.write(f"\nClass Statistics:\n")
        for i, (class_name, paths) in enumerate(list(train_paths.items())[:20]):  # Show first 20 classes
            f.write(f"  {i+1:3d}. {class_name}: {len(paths)} train images\n")
        if len(train_paths) > 20:
            f.write(f"  ... and {len(train_paths) - 20} more classes\n")
    print(f" Saved dataset info to: {dataset_info_path}")
    
    print(f" All files saved to checkpoint directory: {checkpoint_dir}")
    
    return {
        "training_method": "aggressive_hard_negative_mining_full_dataset_straight_through_estimator",
        "dataset": "full_instre",
        "final_loss": hist[-1] if hist else 0.0,
        "avg_final_loss": sum(hist[-10:]) / 10 if len(hist) >= 10 else 0.0,
        "train_classes_count": len(train_paths),
        "total_train_images": sum(len(paths) for paths in train_paths.values()),
        "total_test_images": sum(len(paths) for paths in test_paths.values()),
        "dataset_classes": len(dataset.classes),
        "classes_per_batch": classes_per_batch,
        "samples_per_class": samples_per_class,
        "hard_negative_ratio": hard_negative_ratio,
        "margin": margin,
        "actual_batch_size": actual_batch_size,
        "effective_batch_size": effective_batch_size,
        "straight_through_temperature": temperature,
        "differentiable_buddy_pooling": True,
        "checkpoint_dir": str(checkpoint_dir),
        "timestamp": timestamp
    }


if __name__ == "__main__":
    # Example usage with hard negative mining and straight-through estimator on full dataset
    with app.run():
        # Train with INSANE AGGRESSIVE settings for expert training - STRICTER HARD NEGATIVE MINING
        result = main.remote(
            steps=1000, 
            classes_per_batch=12,  # MAXIMUM classes for hardest negatives
            samples_per_class=6,   # Very few per class = extreme diversity
            hard_negative_ratio=0.02,  # INSANE: Use top 2% hardest negatives only
            margin=0.5,  # EXTREME margin for maximum challenge
            temperature=0.05  # Ultra sharp gradients for precise learning
        )
        print("INSANE AGGRESSIVE Hard Negative Mining + Straight-Through Estimator (Full Dataset) Result:", result)
        
        # Previous AGGRESSIVE settings (now commented)
        # result = main.remote(
        #     steps=1000, 
        #     classes_per_batch=12,  # MORE classes for harder negatives
        #     samples_per_class=6,   # Fewer per class for more diversity
        #     hard_negative_ratio=0.1,  # ULTRA AGGRESSIVE: Use top 10% hardest negatives only
        #     margin=0.3,  # LARGE margin for challenging training
        #     temperature=0.05  # Sharp gradients for better learning
        # )
        
        # Or train with EXTREME AGGRESSIVE settings
        # result = main.remote(
        #     steps=1200, 
        #     classes_per_batch=16,  # MAXIMUM classes for hardest negatives
        #     samples_per_class=4,   # Very few per class = extreme diversity
        #     hard_negative_ratio=0.05,  # EXTREME AGGRESSIVE: Use top 5% hardest negatives only
        #     margin=0.4,  # VERY LARGE margin for extreme challenge
        #     temperature=0.01  # Very sharp gradients
        # ) 