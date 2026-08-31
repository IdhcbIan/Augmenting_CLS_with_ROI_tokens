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
)

# Define Modal App with In-Shop dataset volume
app = modal.App(
    "In-Shop Hard Negative Training",
    image=image,
    volumes={"/mnt/data": modal.Volume.from_name("In-Shop")}
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
        self.backbone_dim = 768  # ViT-B has fixed 768 dimensions
        self.embed_dim = 384     # Our desired output dimension
        self.num_registers = 4
        self.img_size = model_img_size
        self.roi_side = 3
        
        # Create the model - trainable with correct image size
        self.backbone = timm.create_model(MODEL_NAME, pretrained=True, num_classes=0, img_size=self.img_size)
        
        # Add projection layer to transform from backbone_dim to embed_dim
        self.projection = nn.Linear(self.backbone_dim, self.embed_dim)
        
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
        cls_tok = tokens[:, 0:1, :]  # CLS token: (B, 1, 384)
        regs_tok = tokens[:, 1:1 + self.num_registers, :]  # Register tokens: (B, 4, 384)
        patch_tok = tokens[:, 1 + self.num_registers:, :]  # Patch tokens: (B, N, 384)
        
        # Reshape patch tokens to spatial grid
        g = int(self.img_size // 14)  # ViT-B/14 grid size
        patches2d = rearrange(patch_tok, "b (h w) d -> b h w d", h=g, w=g)
        
        # Combine CLS and register tokens to form cues
        cues = torch.cat([cls_tok, regs_tok], dim=1)  # (B, 5, 384)
        
        # Apply buddy pooling to get ROIs with straight-through estimator
        rois = torch.stack([self._buddy_pool(cues[:, i], patches2d)
                           for i in range(cues.size(1))], dim=1)
        
        # Combine cues and ROIs
        toks = torch.cat([cues, rois], dim=1)  # (B, 10, 384)
        
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
            embeddings: (B, 10, 384) - batch embeddings
            labels: (B,) - class labels for each sample (item_id indices)
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
            
            # Find positive samples (same item_id, excluding self)
            pos_mask = (labels == anchor_label) & (torch.arange(batch_size, device=device) != i)
            if not pos_mask.any():
                continue  # Skip if no positives available
                
            # Find negative samples (different item_id)
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


class InShopHardNegativeDataset(Dataset):
    """Dataset for In-Shop hard negative mining with balanced item_id sampling."""
    
    def __init__(self, train_paths, batch_size=32, items_per_batch=8, samples_per_item=4):
        self.batch_size = batch_size
        self.items_per_batch = items_per_batch
        self.samples_per_item = samples_per_item
        
        # train_paths is already a dict of {item_id: [paths]}
        self.item_to_paths = train_paths
        self.items = list(self.item_to_paths.keys())
        
        # Filter items with enough samples
        min_samples = max(2, samples_per_item)
        self.items = [item for item in self.items if len(self.item_to_paths[item]) >= min_samples]
        
        print(f"In-Shop Hard Negative Dataset initialized:")
        print(f"   Total items: {len(self.items)}")
        print(f"   Items per batch: {items_per_batch}")
        print(f"   Samples per item: {samples_per_item}")
        print(f"   Effective batch size: {items_per_batch * samples_per_item}")
        
        # Pre-compute dataset size (arbitrary large number for continuous sampling)
        self.dataset_size = 100000
        
    def __len__(self):
        return self.dataset_size
    
    def __getitem__(self, idx):
        """Sample a balanced batch for hard negative mining."""
        # Select random items for this batch
        selected_items = random.sample(self.items, min(self.items_per_batch, len(self.items)))
        
        batch_images = []
        batch_labels = []
        
        for item_idx, item_id in enumerate(selected_items):
            # Sample images from this item
            item_paths = self.item_to_paths[item_id]
            sampled_paths = random.sample(item_paths, min(self.samples_per_item, len(item_paths)))
            
            for path in sampled_paths:
                image = _load_image(path)
                if image is not None:
                    batch_images.append(image)
                    batch_labels.append(item_idx)  # Use index as label for easier processing
        
        # Pad or truncate to exact batch size if needed
        while len(batch_images) < self.batch_size:
            # Add random samples to fill batch
            random_item = random.choice(selected_items)
            random_path = random.choice(self.item_to_paths[random_item])
            image = _load_image(random_path)
            if image is not None:
                batch_images.append(image)
                batch_labels.append(selected_items.index(random_item))
        
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
        print(f"Warning: Skipping file {path}: {e}")
        return None


def parse_inshop_partitions(root):
    """
    Parse In-Shop evaluation partitions and return train/query/gallery dictionaries by item_id.
    
    Returns:
        train_paths: {item_id: [image_paths]} for training
        query_paths: {item_id: [image_paths]} for query 
        gallery_paths: {item_id: [image_paths]} for gallery
    """
    print("Starting In-Shop partition parsing...")

    # Convert to Path object if string
    root = Path(root)
    partition_file = root / "Eval" / "list_eval_partition.txt"
    if not partition_file.exists():
        raise FileNotFoundError(f"Partition file not found: {partition_file}")

    train_paths = {}
    query_paths = {}
    gallery_paths = {}

    lines = partition_file.read_text().splitlines()
    print(f"Reading {len(lines)} lines from partition file...")

    # Skip header lines (first 2 lines)
    for line_idx, line in enumerate(lines[2:], start=3):
        parts = line.strip().split()
        if len(parts) < 3:
            print(f"Warning: Skipping malformed line {line_idx}: {line}")
            continue
            
        image_name, item_id, status = parts[0], parts[1], parts[2]
        
        # Build full image path (image_name already contains img/ prefix)
        full_image_path = root / image_name
        
        # Check if image exists
        if not full_image_path.exists():
            print(f"Warning: Image not found: {full_image_path}")
            continue
        
        # Add to appropriate partition
        if status == "train":
            if item_id not in train_paths:
                train_paths[item_id] = []
            train_paths[item_id].append(str(full_image_path))
        elif status == "query":
            if item_id not in query_paths:
                query_paths[item_id] = []
            query_paths[item_id].append(str(full_image_path))
        elif status == "gallery":
            if item_id not in gallery_paths:
                gallery_paths[item_id] = []
            gallery_paths[item_id].append(str(full_image_path))
    
    # Filter items that have sufficient samples for training
    train_paths = {item: paths for item, paths in train_paths.items() if len(paths) >= 2}
    
    print(f"In-Shop Dataset Statistics:")
    print(f"   - Train items: {len(train_paths)}")
    train_sizes = [len(paths) for paths in train_paths.values()]
    if train_sizes:
        print(f"   - Train images per item: min={min(train_sizes)}, max={max(train_sizes)}, avg={sum(train_sizes)/len(train_sizes):.1f}")
        print(f"   - Total train images: {sum(train_sizes)}")

    print(f"   - Query items: {len(query_paths)}")
    query_sizes = [len(paths) for paths in query_paths.values()]
    if query_sizes:
        print(f"   - Total query images: {sum(query_sizes)}")

    print(f"   - Gallery items: {len(gallery_paths)}")
    gallery_sizes = [len(paths) for paths in gallery_paths.values()]
    if gallery_sizes:
        print(f"   - Total gallery images: {sum(gallery_sizes)}")
    
    return train_paths, query_paths, gallery_paths


@app.function(
    gpu="A100-80GB:2",  # 2 A100-80GB GPUs
    timeout=8500  # 2.5 hour timeout
)
def main(
    inshop_root: str = "/mnt/data",
    steps: int = 2000,
    batch_size: int = 64,   # Smaller batch for hard negative mining
    report_interval: int = 1,
    lr: float = 2e-6,
        items_per_batch: int = 12,  # Number of different items in each batch
    samples_per_item: int = 6,   # Number of samples per item in each batch
    hard_negative_ratio: float = 0.1,  # Fraction of hardest negatives to use
    margin: float = 0.2,  # Triplet loss margin
    temperature: float = 0.05  # Temperature for straight-through estimator
):
    """
    Train In-Shop model using hard negative mining on item_id-based instance retrieval
    with straight-through estimator for differentiable buddy pooling.
    
    Args:
        inshop_root: Path to In-Shop dataset root
        steps: Number of training steps
        batch_size: Effective batch size (items_per_batch * samples_per_item)
        report_interval: Steps between progress reports
        lr: Learning rate
        items_per_batch: Number of different items in each batch
        samples_per_item: Number of samples per item in each batch
        hard_negative_ratio: Fraction of hardest negatives to use (0.0-1.0)
        margin: Triplet loss margin
        temperature: Temperature for straight-through estimator (lower = sharper gradients)
    """
    print(f"Starting In-Shop Hard Negative Mining Training with Straight-Through Estimator")
    print(f"   Dataset: In-Shop Clothes Retrieval Benchmark")
    print(f"   Task: INSTANCE-LEVEL retrieval (item_id-based)")
    print(f"   Hard Negative Mining: {items_per_batch} items × {samples_per_item} samples = {items_per_batch * samples_per_item} per batch")
    print(f"   Hard negative ratio: {hard_negative_ratio} (top {int(hard_negative_ratio*100)}% hardest)")
    print(f"   Margin: {margin}")
    print(f"   Straight-through estimator temperature: {temperature}")
    print("="*60)
    
    # Setup multi-GPU environment
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
    
    inshop_root_path = Path(inshop_root)
    
    # Load In-Shop dataset partitions
    train_paths, query_paths, gallery_paths = parse_inshop_partitions(inshop_root_path)
    
    if not train_paths:
        raise ValueError("No train files found in In-Shop dataset")
    
    print(f"In-Shop Dataset Loaded:")
    print(f"   Training items: {len(train_paths)}")
    print(f"   Total train images: {sum(len(paths) for paths in train_paths.values())}")
    print(f"   Query items: {len(query_paths)}")
    print(f"   Gallery items: {len(gallery_paths)}")
    
    # Setup multi-GPU
    num_gpus = torch.cuda.device_count()
    print(f"Using {num_gpus} GPUs: {[torch.cuda.get_device_name(i) for i in range(num_gpus)]}")
    
    # Create model with straight-through estimator
    print("Creating model with differentiable buddy pooling (straight-through estimator)")
    model = TrainableMultiVectorEncoder()
    
    if num_gpus > 1:
        device = torch.device("cuda:0")
        model = model.to(device)
        model = nn.DataParallel(model)
        print(f"Model wrapped with DataParallel across {num_gpus} GPUs")
        effective_batch_size = batch_size * num_gpus
    else:
        device = torch.device("cuda:0")
        model = model.to(device)
        effective_batch_size = batch_size
    
    print(f"Model has {sum(p.numel() for p in model.parameters() if p.requires_grad):,} trainable parameters")
    
    # Calculate actual effective batch size for hard negative mining
    actual_batch_size = items_per_batch * samples_per_item
    if num_gpus > 1:
        effective_batch_size = actual_batch_size * num_gpus
    else:
        effective_batch_size = actual_batch_size
        
    print(f"Hard Negative Mining Configuration:")
    print(f"   Items per batch: {items_per_batch}")
    print(f"   Samples per item: {samples_per_item}")
    print(f"   Actual batch size: {actual_batch_size}")
    print(f"   Effective batch size: {effective_batch_size} (gpus: {num_gpus})")
    print(f"   Hard negative ratio: {hard_negative_ratio} (top {int(hard_negative_ratio*100)}% hardest)")
    print(f"   Margin: {margin}")
    print(f"   Learning rate: {lr} (constant - no scheduler)")
    print(f"   Straight-through temperature: {temperature}")
    
    # Setup optimizer and hard negative mining loss (constant LR)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    criterion = HardNegativeTripletLoss(margin=margin, hard_negative_ratio=hard_negative_ratio)
    
    # Setup dataset and dataloader for hard negative mining
    print(" Setting up Hard Negative Mining Dataset...")
    dataset = InShopHardNegativeDataset(
        train_paths, 
        batch_size=actual_batch_size,
        items_per_batch=items_per_batch,
        samples_per_item=samples_per_item
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
    
    print(f"Hard Negative Dataset ready:")
    print(f"   Total items: {len(dataset.items)}")
    print(f"   Dataset steps: {len(dataset)}")
    print(f"   Dataloader workers: {num_workers}")
    
    # Training loop with hard negative mining and straight-through estimator
    hist = []
    
    for i, batch in enumerate(tqdm(dataloader, desc="In-Shop Hard Negative Mining Training (Straight-Through)", total=steps)):
        if i >= steps:
            break
            
        # Unpack batch (dataset returns (images, labels) as a tuple)
        images, labels = batch[0].squeeze(0), batch[1].squeeze(0)  # Remove dataloader batch dimension
        images, labels = images.to(device), labels.to(device)
        
        print(f"Step {i+1}/{steps}: Processing batch with {len(torch.unique(labels))} items...")
        print(f"   Batch shape: {images.shape}, Labels: {labels.shape}")
        print(f"   Using differentiable buddy pooling (straight-through estimator)")

        optimizer.zero_grad()
        
        # Forward pass to get embeddings (now with differentiable buddy pooling)
        embeddings = model(images)  # (B, 10, 384)
        
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
            print(f"[step {i+1:4d}] avg loss: {avg_loss:.4f} (hard_neg_ratio: {hard_negative_ratio}, margin: {margin})")

    print("In-Shop Training completed!")
    print(f"Final loss: {hist[-1]:.6f}")

    # Save model checkpoint
    print("Saving model checkpoint...")

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    checkpoint_dir = Path(f"/mnt/data/Checkpoints/inshop_hardneg_estimator_{timestamp}")
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    print(f"Created checkpoint directory: {checkpoint_dir}")
    
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
            'items_per_batch': items_per_batch,
            'samples_per_item': samples_per_item,
            'hard_negative_ratio': hard_negative_ratio,
            'margin': margin,
            'lr': lr,
            'model_img_size': model_img_size,
            'num_gpus': num_gpus,
            'train_items_count': len(train_paths),
            'total_train_images': sum(len(paths) for paths in train_paths.values()),
            'query_items_count': len(query_paths),
            'gallery_items_count': len(gallery_paths),
            'dataset_items': len(dataset.items),
            'training_method': 'aggressive_hard_negative_mining_inshop_instance_retrieval_straight_through_estimator',
            'straight_through_temperature': temperature,
            'differentiable_buddy_pooling': True,
            'learning_rate_schedule': 'constant',
            'dataset': 'inshop_clothes_retrieval'
        },
        'timestamp': timestamp
    }
    
    checkpoint_path = checkpoint_dir / "checkpoint.pth"
    torch.save(checkpoint, checkpoint_path)
    print(f"Saved complete checkpoint to: {checkpoint_path}")

    # Save model weights
    model_weights_path = checkpoint_dir / "model_weights.pth"
    torch.save(model_to_save.state_dict(), model_weights_path)
    print(f"Saved model weights to: {model_weights_path}")
    
    # Save configuration
    config_path = checkpoint_dir / "config.txt"
    with open(config_path, 'w') as f:
        f.write(f"In-Shop Hard Negative Mining Training - Straight-Through Estimator - {timestamp}\n")
        f.write("=" * 80 + "\n")
        f.write(f"Training Method: Hard Negative Mining + Straight-Through Estimator\n")
        f.write(f"Dataset: In-Shop Clothes Retrieval Benchmark\n")
        f.write(f"Task: INSTANCE-LEVEL retrieval (item_id-based)\n")
        f.write(f"Steps: {steps}\n")
        f.write(f"Batch Configuration:\n")
        f.write(f"  - Items per batch: {items_per_batch}\n")
        f.write(f"  - Samples per item: {samples_per_item}\n")
        f.write(f"  - Actual batch size: {actual_batch_size}\n")
        f.write(f"  - Effective batch size: {effective_batch_size}\n")
        f.write(f"Hard Negative Mining:\n")
        f.write(f"  - Hard negative ratio: {hard_negative_ratio} (top {int(hard_negative_ratio*100)}%)\n")
        f.write(f"  - Triplet margin: {margin}\n")
        f.write(f"Straight-Through Estimator:\n")
        f.write(f"  - Temperature: {temperature}\n")
        f.write(f"  - Differentiable buddy pooling: Enabled\n")
        f.write(f"  - Forward pass: Discrete argmax selection\n")
        f.write(f"  - Backward pass: Softmax with low temperature\n")
        f.write(f"Learning Rate: {lr} (constant - no scheduler)\n")
        f.write(f"Model Image Size: {model_img_size}\n")
        f.write(f"Number of GPUs: {num_gpus}\n")
        f.write(f"Final Loss: {hist[-1] if hist else 0.0:.6f}\n")
        f.write(f"Training Items: {len(train_paths)}\n")
        f.write(f"Total Train Images: {sum(len(paths) for paths in train_paths.values())}\n")
        f.write(f"Query Items: {len(query_paths)}\n")
        f.write(f"Gallery Items: {len(gallery_paths)}\n")
        f.write(f"Dataset Items: {len(dataset.items)}\n")
        f.write(f"Total Training Steps: {len(hist)}\n")
    print(f"Saved training config to: {config_path}")

    # Save dataset info for reference
    dataset_info_path = checkpoint_dir / "dataset_info.txt"
    with open(dataset_info_path, 'w') as f:
        f.write(f"In-Shop Dataset Information\n")
        f.write("=" * 50 + "\n")
        f.write(f"Training Items: {len(train_paths)}\n")
        f.write(f"Total Train Images: {sum(len(paths) for paths in train_paths.values())}\n")
        f.write(f"Query Items: {len(query_paths)}\n")
        f.write(f"Gallery Items: {len(gallery_paths)}\n")
        f.write(f"\nStraight-Through Estimator Details:\n")
        f.write(f"Temperature: {temperature}\n")
        f.write(f"Forward Pass: Discrete argmax selection for buddy pooling\n")
        f.write(f"Backward Pass: Softmax with temperature {temperature} for gradients\n")
        f.write(f"Benefit: Enables gradient flow through buddy pooling operation\n")
        f.write(f"\nItem Statistics:\n")
        item_sizes = [len(paths) for paths in train_paths.values()]
        f.write(f"Images per item: min={min(item_sizes)}, max={max(item_sizes)}, avg={sum(item_sizes)/len(item_sizes):.1f}\n")
        for i, (item_id, paths) in enumerate(list(train_paths.items())[:20]):  # Show first 20 items
            f.write(f"  {i+1:3d}. Item {item_id}: {len(paths)} train images\n")
        if len(train_paths) > 20:
            f.write(f"  ... and {len(train_paths) - 20} more items\n")
    print(f"Saved dataset info to: {dataset_info_path}")

    print(f"All files saved to checkpoint directory: {checkpoint_dir}")
    
    return {
        "training_method": "hard_negative_mining_inshop_instance_retrieval_straight_through_estimator",
        "dataset": "inshop_clothes_retrieval",
        "final_loss": hist[-1] if hist else 0.0,
        "avg_final_loss": sum(hist[-10:]) / 10 if len(hist) >= 10 else 0.0,
        "train_items_count": len(train_paths),
        "total_train_images": sum(len(paths) for paths in train_paths.values()),
        "query_items_count": len(query_paths),
        "gallery_items_count": len(gallery_paths),
        "dataset_items": len(dataset.items),
        "items_per_batch": items_per_batch,
        "samples_per_item": samples_per_item,
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
    # Example usage with hard negative mining and straight-through estimator
    with app.run():
        result = main.remote(
            steps=1000,
            items_per_batch=12,
            samples_per_item=6,
            hard_negative_ratio=0.02,
            margin=0.5,
            temperature=0.05
        )
        print("In-Shop Hard Negative Mining + Straight-Through Estimator Result:", result) 
