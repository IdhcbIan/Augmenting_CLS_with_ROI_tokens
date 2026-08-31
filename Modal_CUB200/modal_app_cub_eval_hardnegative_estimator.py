import modal
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
import numpy as np
from datetime import datetime
from torch.utils.data import Dataset, DataLoader


# Build Modal Image including local Python source code
image = (
    modal.Image.debian_slim()
    .pip_install("torch", "torchvision", "tqdm", "timm==0.9.12", "einops==0.7.0", "pillow", "numpy")
    .add_local_file("buddy_pool.py", "/root/buddy_pool.py")
    .add_local_file("maxsim_loss.py", "/root/maxsim_loss.py")
)

# Define Modal App with dataset volume
app = modal.App(
    "CUB EVAL HARDNEGATIVE ESTIMATOR PADDING AND SCALING",
    image=image,
    volumes={"/mnt/data": modal.Volume.from_name("cub-data")}
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
    """TRAINABLE Multi-vector encoder with straight-through estimator (same as training script)."""
    
    def __init__(self):
        super().__init__()
        MODEL_NAME = "vit_base_patch14_reg4_dinov2.lvd142m"
        
        # Configuration matching training script
        self.backbone_dim = 768  # ViT-B has fixed 768 dimensions
        self.embed_dim = 384     # Our desired output dimension
        self.num_registers = 4
        self.img_size = model_img_size
        self.roi_side = 9
        self.patch_size = 14
        
        # Create the model - trainable with correct image size
        self.backbone = timm.create_model(MODEL_NAME, pretrained=True, num_classes=0, img_size=self.img_size)
        
        # Add projection layer to transform from backbone_dim to embed_dim
        self.projection = nn.Linear(self.backbone_dim, self.embed_dim)
        
    def _buddy_pool(self, cue, patches2d, valid_masks=None):
        """Improved buddy pooling with straight-through estimator and optional valid masks."""
        B, H, W, d = patches2d.shape
        flat = rearrange(patches2d, "b h w d -> b (h w) d")
        
        if valid_masks is None:
            sim = torch.matmul(cue.unsqueeze(1), flat.transpose(1, 2)).squeeze(1)
            # Use straight-through estimator: argmax forward, softmax backward
            idx = StraightThroughArgmax.apply(sim, 0.05)  # temperature = 0.05 for sharper gradients
        else:
            valid_flat = rearrange(valid_masks, "b h w -> b (h w)")
            sim = torch.matmul(cue.unsqueeze(1), flat.transpose(1, 2)).squeeze(1)
            sim = sim.masked_fill(~valid_flat, float("-inf"))
            # Use straight-through estimator even with masks
            idx = StraightThroughArgmax.apply(sim, 0.05)
            
        h = idx // W
        w = idx % W
        r = self.roi_side // 2
        roi = []
        
        for b in range(B):
            hb = (idx[b] // W).item()
            wb = (idx[b] % W).item()
            hs = slice(max(0, hb - r), min(H, hb + r + 1))
            ws = slice(max(0, wb - r), min(W, wb + r + 1))
            roi_patches = patches2d[b, hs, ws, :]
            
            if valid_masks is None:
                pooled = roi_patches.mean(dim=(0, 1))
            else:
                valid_in_roi = valid_masks[b, hs, ws]
                num_valid = valid_in_roi.sum()
                if num_valid > 0:
                    pooled = torch.sum(roi_patches * valid_in_roi.unsqueeze(-1).float(), dim=(0, 1)) / num_valid.float()
                else:
                    pooled = torch.zeros(d, device=patches2d.device)
            roi.append(pooled)
            
        return torch.stack(roi)
        
    def forward(self, x, valid_masks=None):
        """Clean forward pass with differentiable buddy pooling and optional valid masks."""
        # Forward through backbone
        tokens = self.backbone.forward_features(x)
        
        # Apply projection (trainable layer)
        tokens = self.projection(tokens)
        
        # Extract tokens following original structure
        cls_tok = tokens[:, 0:1, :]  # CLS token: (B, 1, D)
        regs_tok = tokens[:, 1:1 + self.num_registers, :]  # Register tokens: (B, 4, D)
        patch_tok = tokens[:, 1 + self.num_registers:, :]  # Patch tokens: (B, N, D)
        
        # Reshape patch tokens to spatial grid
        g = int(self.img_size // self.patch_size)  # ViT-B/14 grid size
        patches2d = rearrange(patch_tok, "b (h w) d -> b h w d", h=g, w=g)
        
        # Combine CLS and register tokens to form cues
        cues = torch.cat([cls_tok, regs_tok], dim=1)  # (B, 5, D)
        
        # Apply buddy pooling to get ROIs with straight-through estimator
        rois = torch.stack([self._buddy_pool(cues[:, i], patches2d, valid_masks)
                           for i in range(cues.size(1))], dim=1)
        
        # Combine cues and ROIs
        toks = torch.cat([cues, rois], dim=1)  # (B, 10, D)
        
        # Normalize
        return F.normalize(toks, dim=-1)


#------// Dataset Class for Gallery Building //-----------------

class GalleryDataset(Dataset):
    """Dataset for efficient gallery building with DataLoader."""
    
    def __init__(self, file_list, cub_root):
        self.file_list = file_list  # List of (file_path, class_name) tuples
        self.cub_root = Path(cub_root)
        self.transform = transforms.Compose([
            transforms.Resize(224),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                               std=[0.229, 0.224, 0.225]),
        ])
    
    def __len__(self):
        return len(self.file_list)
    
    def __getitem__(self, idx):
        file_path, class_name = self.file_list[idx]
        
        try:
            img = Image.open(file_path).convert("RGB")
            img_tensor = self.transform(img)
            return img_tensor, class_name, str(file_path)
        except (FileNotFoundError, OSError) as e:
            print(f"Warning: Error loading {file_path}: {e}")
            # Return a dummy tensor
            dummy_tensor = torch.zeros(3, 224, 224)
            return dummy_tensor, class_name, str(file_path)


#------// Model Params //-----------------

model_img_size = 224


#------// Efficient Multi-Scale Query Dataset //-----------------

class EfficientMultiScaleQueryDataset(Dataset):
    """EFFICIENT dataset that processes ALL padding scales for each query in a single batch."""
    
    def __init__(self, query_list, padding_scales=[0.6, 0.8, 1.0, 1.2, 1.4], valid_threshold=0.5):
        self.query_list = query_list  # List of (query_path, class_name) tuples
        self.padding_scales = padding_scales
        self.valid_threshold = valid_threshold
        self.num_scales = len(padding_scales)
        
        print(f"Efficient Multi-Scale Dataset initialized:")
        print(f"   Total queries: {len(query_list)}")
        print(f"   Padding scales: {padding_scales}")
        print(f"   Scales per query: {self.num_scales}")
        print(f"   Total scale-query combinations: {len(query_list) * self.num_scales}")
    
    def __len__(self):
        return len(self.query_list)
    
    def __getitem__(self, idx):
        query_path, class_name = self.query_list[idx]
        
        # Process ALL scales for this query efficiently
        all_scales_tensors = []
        all_scales_masks = []
        
        for padding_scale in self.padding_scales:
            img_tensor, valid_mask = self._apply_padding_and_mask(query_path, padding_scale)
            all_scales_tensors.append(img_tensor)
            all_scales_masks.append(valid_mask)
        
        # Stack all scales for this query: (num_scales, 3, 224, 224) and (num_scales, 16, 16)
        stacked_tensors = torch.stack(all_scales_tensors)  # (num_scales, 3, 224, 224)
        stacked_masks = torch.stack(all_scales_masks)      # (num_scales, 16, 16)
        
        return stacked_tensors, stacked_masks, class_name, query_path
    
    def _apply_padding_and_mask(self, image_path, padding_scale, target_size=224):
        """Apply padding to image and create valid mask."""
        try:
            # Load original image
            img = Image.open(image_path).convert("RGB")
            orig_w, orig_h = img.size
            
            # Calculate new size based on padding scale
            new_w = int(orig_w * padding_scale)
            new_h = int(orig_h * padding_scale)
            
            # Resize image to new size
            img_resized = img.resize((new_w, new_h), Image.LANCZOS)
            
            # Create target-sized canvas and paste centered
            canvas = Image.new("RGB", (target_size, target_size), (0, 0, 0))  # Black padding
            paste_x = (target_size - new_w) // 2
            paste_y = (target_size - new_h) // 2
            canvas.paste(img_resized, (paste_x, paste_y))
            
            # Apply transforms
            transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])
            img_tensor = transform(canvas)
            
            # Create valid mask
            valid_mask = self._create_valid_mask(padding_scale, target_size)
            
            return img_tensor, valid_mask
            
        except Exception as e:
            print(f"Warning: Error processing {image_path}: {e}")
            # Return dummy data
            dummy_tensor = torch.zeros(3, target_size, target_size)
            dummy_mask = torch.ones(16, 16, dtype=torch.bool)  # 224//14 = 16
            return dummy_tensor, dummy_mask
    
    def _create_valid_mask(self, padding_scale, img_size=224, patch_size=14):
        """Create valid mask for padded image."""
        new_size = int(img_size * padding_scale)
        start = (img_size - new_size) // 2
        end = start + new_size
        
        # Create mask for valid region
        mask = torch.zeros(img_size, img_size, dtype=torch.bool)
        mask[start:end, start:end] = True
        
        # Downsample to patch grid size
        grid_size = img_size // patch_size
        mask_patches = F.avg_pool2d(mask.float().unsqueeze(0).unsqueeze(0), 
                                   kernel_size=patch_size, stride=patch_size).squeeze()
        
        return mask_patches > 0.5


def load_model_checkpoint(checkpoint_path, device='cuda:0'):
    """Load a saved model checkpoint."""
    if Path(checkpoint_path).is_dir():
        checkpoint_path = Path(checkpoint_path) / "checkpoint.pth"
    
    print(f"Loading checkpoint from: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location=device)

    # Create model and load weights
    model = TrainableMultiVectorEncoder()
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    model.eval()

    print(f"Model loaded successfully!")
    print(f"   - Timestamp: {checkpoint.get('timestamp', 'Unknown')}")
    print(f"   - Final Loss: {checkpoint.get('final_loss', 'Unknown')}")
    
    # Print training config if available
    if 'training_config' in checkpoint:
        config = checkpoint['training_config']
        print(f"   - Training method: {config.get('training_method', 'Unknown')}")
        print(f"   - Training steps: {config.get('steps', 'Unknown')}")
        print(f"   - Hard negative ratio: {config.get('hard_negative_ratio', 'Unknown')}")
        print(f"   - Margin: {config.get('margin', 'Unknown')}")
        print(f"   - Straight-through temp: {config.get('straight_through_temperature', 'Unknown')}")
        print(f"   - Total train images: {config.get('total_train_images', 'Unknown')}")
        print(f"   - Dataset classes: {config.get('dataset_classes', 'Unknown')}")
    
    return model, checkpoint


def parse_cub_for_eval(root: Path):
    """Parse CUB dataset and return training/test image path dictionaries for evaluation."""
    print("Starting CUB parsing for evaluation...")
    
    cls_map = {}
    for line in (root / "classes.txt").read_text().splitlines():
        cid, cname = line.split()
        cls_map[int(cid)] = cname

    img_to_cid = {}
    for line in (root / "image_class_labels.txt").read_text().splitlines():
        iid, cid = line.split()
        img_to_cid[int(iid)] = int(cid)

    img_map = {}
    for line in (root / "images.txt").read_text().splitlines():
        iid, rel = line.split()
        img_map[int(iid)] = "/mnt/data/CUB_200_2011/images/" + rel

    train_ids = set()
    for line in (root / "train_test_split.txt").read_text().splitlines():
        iid, flag = line.split()
        if int(flag):
            train_ids.add(int(iid))

    train_files = []
    test_files = []
    
    for iid, path in img_map.items():
        cname = cls_map[img_to_cid[iid]]
        if iid in train_ids:
            train_files.append((path, cname))
        else:
            test_files.append((path, cname))
    
    print(f"CUB Dataset Statistics for Evaluation:")
    train_classes = set(class_name for _, class_name in train_files)
    test_classes = set(class_name for _, class_name in test_files)
    print(f"   - Train files: {len(train_files)} ({len(train_classes)} classes)")
    print(f"   - Test files: {len(test_files)} ({len(test_classes)} classes)")
    print(f"   - Common classes: {len(train_classes & test_classes)}")
    
    return train_files, test_files


def maxsim_scorer(q, g, chunk=2048):
    """ColBERT MaxSim scorer with chunking for memory efficiency."""
    sims_chunks = []
    for g_chunk in torch.split(g, chunk, dim=0):
        s = torch.einsum('btd,gkd->btgk', q, g_chunk)
        s = s.max(dim=-1).values
        s = s.sum(dim=1)
        sims_chunks.append(s)
    return torch.cat(sims_chunks, dim=1)


def evaluate_retrieval_cub_dataset(model, checkpoint_dir, cub_root, eval_batch_size, device, ks=[1, 2, 4, 8], valid_threshold=0.5, padding_scales=[0.6, 0.8, 1.0, 1.2, 1.4]):
    """Efficient evaluation: Process ALL padding scales in parallel batches for maximum speed."""
    print(f"\nEfficient CUB-200-2011 evaluation with parallel multi-scale processing")
    print("="*50)
    
    # Load CUB dataset
    train_files, test_files = parse_cub_for_eval(Path(cub_root))
    
    if not train_files:
        print(f"No train files found in CUB dataset")
        return None

    # Build gallery from train files using DataLoader (same as before - this is already efficient)
    print(f"Creating gallery dataset from {len(train_files)} train files...")
    gallery_dataset = GalleryDataset(train_files, cub_root)
    gallery_dataloader = DataLoader(
        gallery_dataset,
        batch_size=eval_batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True
    )
    
    gallery_embeddings = []
    gallery_labels = []
    gallery_items = []
    
    # Process train files efficiently with DataLoader
    model.eval()
    with torch.no_grad():
        for batch_imgs, batch_labels, batch_paths in tqdm(gallery_dataloader, desc="Building gallery"):
            # Move batch to device
            batch_imgs = batch_imgs.to(device)
            
            # Get embeddings for the batch
            embs = model(batch_imgs)  # (B, 10, 384)
            
            # Store embeddings and metadata
            for emb, label, path in zip(embs, batch_labels, batch_paths):
                gallery_embeddings.append(emb.cpu())
                gallery_labels.append(label)
                gallery_items.append((label, path))
    
    if not gallery_embeddings:
        print("No valid gallery embeddings created")
        return None
    
    gallery_stack = torch.stack(gallery_embeddings).to(device)  # (G, 10, 384)
    gallery_stack = F.normalize(gallery_stack, dim=-1)
    
    print(f"Gallery built: {gallery_stack.shape}")
    print(f"Gallery classes: {len(set(gallery_labels))} unique classes")
    print(f"Gallery items: {len(gallery_items)} total items")
    
    if not test_files:
        print(f"No test files found in CUB dataset")
        return None
    
    # Efficient: Process ALL scales in parallel using the new dataset
    print(f"\nEfficient processing of ALL scales in parallel: {padding_scales}")
    print(f"   Strategy: Process all scales per query in single batches")
    
    hits = {k: 0 for k in ks}
    aps = []
    total_queries = len(test_files)
    num_scales = len(padding_scales)
    
    # Create EFFICIENT multi-scale query dataset
    query_dataset = EfficientMultiScaleQueryDataset(test_files, padding_scales, valid_threshold)
    query_dataloader = DataLoader(
        query_dataset, 
        batch_size=eval_batch_size // 2,  # Smaller batch since each item contains multiple scales
        shuffle=False, 
        num_workers=4,  # More workers since we're doing more work per item
        pin_memory=True
    )
    
    # Process all queries with all scales efficiently
    all_query_embeddings = []  # Store (query_embs_all_scales, query_label) for each query
    query_labels_list = []
    
    print(f"Processing {total_queries} queries with {num_scales} scales each...")
    
    model.eval()
    with torch.no_grad():
        for batch_stacked_imgs, batch_stacked_masks, batch_labels, batch_paths in tqdm(query_dataloader, desc="Processing multi-scale queries"):
            batch_size = batch_stacked_imgs.size(0)
            
            # batch_stacked_imgs: (B, num_scales, 3, 224, 224)
            # batch_stacked_masks: (B, num_scales, 16, 16)
            
            # Reshape to process all scales in one forward pass
            # (B * num_scales, 3, 224, 224) and (B * num_scales, 16, 16)
            flat_imgs = batch_stacked_imgs.view(-1, 3, 224, 224).to(device)
            flat_masks = batch_stacked_masks.view(-1, 16, 16).to(device)
            
            # Forward pass for all scales at once
            all_embs = model(flat_imgs, flat_masks)  # (B * num_scales, 10, 384)
            all_embs = F.normalize(all_embs, dim=-1)
            
            # Reshape back to (B, num_scales, 10, 384)
            reshaped_embs = all_embs.view(batch_size, num_scales, 10, 384)
            
            # Store embeddings and labels
            for query_embs, label in zip(reshaped_embs, batch_labels):
                all_query_embeddings.append(query_embs.cpu())  # (num_scales, 10, 384)
                query_labels_list.append(label)
    
    print(f"Processed all queries efficiently: {len(all_query_embeddings)} queries × {num_scales} scales")
    
    # Now compute similarities and metrics efficiently
    print("Computing MAX similarities across scales for all queries...")
    
    for query_idx, (query_embs_all_scales, query_label) in enumerate(tqdm(zip(all_query_embeddings, query_labels_list), total=total_queries, desc="Computing similarities")):
        # query_embs_all_scales: (num_scales, 10, 384)
        query_embs_all_scales = query_embs_all_scales.to(device)
        
        # Compute similarities for all scales at once
        query_sims_all_scales = maxsim_scorer(query_embs_all_scales, gallery_stack)  # (num_scales, G)
        
        # Take MAX across scales
        combined_sim = query_sims_all_scales.max(dim=0).values  # (G,)
        
        # Compute topk for recall metrics
        topk_indices = combined_sim.topk(max(ks), dim=0).indices.cpu()
        
        # Recall@k
        for k in ks:
            if any(gallery_items[idx][0] == query_label for idx in topk_indices[:k]):
                hits[k] += 1
        
        # Average Precision (AP)
        sorted_indices = torch.argsort(combined_sim, descending=True).cpu()
        
        # Compute AP with full gallery depth
        MAP_DEPTH = len(gallery_items)
        relevances = [1 if gallery_items[idx][0] == query_label else 0 
                    for idx in sorted_indices[:MAP_DEPTH]]
        
        num_rel, ap_sum = 0, 0.0
        for rank, rel in enumerate(relevances, start=1):
            if rel:
                num_rel += 1
                ap_sum += num_rel / rank
        aps.append(ap_sum / num_rel if num_rel > 0 else 0.0)
    
    # Compute final metrics
    final_results = {}
    for k in ks:
        recall = hits[k] / total_queries if total_queries > 0 else 0.0
        final_results[f'recall@{k}'] = recall
    
    mAP = sum(aps) / len(aps) if aps else 0.0
    final_results['mAP'] = mAP
    final_results['total_queries'] = total_queries
    
    # Print final summary
    print("\n" + "="*60)
    print("Efficient CUB-200-2011 RETRIEVAL EVALUATION RESULTS")
    print("   (Hardnegative Estimator with Parallel Multi-Scale Processing)")
    print("="*60)
    print(f"Strategy: Parallel processing + MAX combination across {len(padding_scales)} scales")
    print(f"Total queries processed: {total_queries}")
    print(f"Efficiency: ALL scales processed in parallel batches")
    for k in ks:
        print(f"   Recall@{k}: {final_results[f'recall@{k}']:.4f} ({hits[k]}/{total_queries})")
    print(f"   mAP: {mAP:.4f}")
    print("="*60)
    
    # Save detailed results
    results_file = checkpoint_dir / f"eval_results_cub_max_combination_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
    with open(results_file, 'w') as f:
        f.write(f"CUB-200-2011 Evaluation Results - Hardnegative Estimator (MAX Combination)\n")
        f.write(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write("="*60 + "\n")
        f.write(f"Model: CUB Hard Negative Mining + Straight-Through Estimator\n")
        f.write(f"Dataset: CUB-200-2011\n")
        f.write(f"Strategy: MAX combination across scales {padding_scales}\n")
        f.write(f"Total Gallery Items: {len(gallery_items)}\n")
        f.write(f"Total Query Items: {total_queries}\n\n")
        
        f.write("Combined Results (MAX across all scales):\n")
        for k in ks:
            f.write(f"  Recall@{k}: {final_results[f'recall@{k}']:.4f}\n")
        f.write(f"  mAP: {mAP:.4f}\n")
    
    print(f"Results saved to: {results_file}")
    
    return final_results


@app.function(
    gpu="A100-80GB:1",  # Single A100 for evaluation
    timeout=2400  # 40 minutes timeout
)
def main(
    checkpoint_path: str = "/mnt/data/Checkpoints/cub_aggressive_hardneg_estimator_20250826_201554",
    cub_root: str = "/mnt/data/CUB_200_2011",
    eval_batch_size: int = 128,
    valid_threshold: float = 0.5,  # Valid mask threshold for padding
    padding_scales = [0.6, 0.8, 1.0, 1.2, 1.4],  # Multiple padding scales
):
    """
    Efficient evaluation of CUB Hardnegative Estimator model using parallel multi-scale processing.

    Speed improvements:
    - Processes ALL padding scales in parallel batches (not sequentially)
    - Single forward pass for all scales per query batch
    - Efficient tensor reshaping and batching

    Args:
        checkpoint_path: Path to model checkpoint (directory or .pth file)
        cub_root: Path to CUB_200_2011 directory
        eval_batch_size: Batch size for evaluation
        valid_threshold: Threshold for valid mask computation
        padding_scales: List of padding scales to apply to queries (MAX combination used)
    """
    print(f"Starting Efficient CUB Hardnegative Estimator Model Evaluation")
    print(f"   Checkpoint: {checkpoint_path}")
    print(f"   Dataset: CUB-200-2011")
    print(f"   Padding scales: {padding_scales} (parallel processing + MAX combination)")
    print(f"   Efficiency: ALL scales processed in parallel batches")
    print("="*60)
    
    # Setup device
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Load model
    model, checkpoint = load_model_checkpoint(checkpoint_path, device)
    
    # Find checkpoint directory for saving results
    if Path(checkpoint_path).is_dir():
        checkpoint_dir = Path(checkpoint_path)
    else:
        checkpoint_dir = Path(checkpoint_path).parent
    
    # Evaluate on CUB dataset
    print(f"\n{'='*20} CUB-200-2011 with MULTIPLE PADDINGS {'='*20}")

    try:
        results = evaluate_retrieval_cub_dataset(
            model, checkpoint_dir, cub_root, eval_batch_size, device,
            ks=[1, 2, 4, 8], valid_threshold=valid_threshold, padding_scales=padding_scales
        )

        if results is None:
            print("Evaluation failed")
            return {"status": "failed", "error": "evaluation_failed"}

        print("Evaluation completed successfully!")

        return {
            "status": "success",
            "checkpoint_path": checkpoint_path,
            "dataset": "cub_200_2011",
            "padding_scales": padding_scales,
            "combination_strategy": "max",
            "results": results,
            "timestamp": datetime.now().strftime('%Y%m%d_%H%M%S')
        }

    except Exception as e:
        print(f"Error during evaluation: {e}")
        import traceback
        traceback.print_exc()
        return {"status": "error", "error": str(e)}


if __name__ == "__main__":
    # Example usage
    with app.run():
        result = main.remote(
            checkpoint_path="/mnt/data/Checkpoints/cub_aggressive_hardneg_estimator_20250730_123252",
            eval_batch_size=128,
            padding_scales=[0.4, 0.6, 0.8, 1.0, 1.2, 1.4, 1.6]
        )
        print("Efficient CUB Evaluation Result:", result)
        
