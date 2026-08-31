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

# Define Modal App with In-Shop dataset volume
app = modal.App(
    "In-Shop Instance Retrieval Evaluation",
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
    """TRAINABLE Multi-vector encoder with straight-through estimator (same as training script)."""
    
    def __init__(self):
        super().__init__()
        MODEL_NAME = "vit_base_patch14_reg4_dinov2.lvd142m"
        
        # Configuration matching original
        self.backbone_dim = 768  # ViT-B has fixed 768 dimensions
        self.embed_dim = 384     # Our desired output dimension
        self.num_registers = 4
        self.img_size = model_img_size
        self.roi_side = 3
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
        cls_tok = tokens[:, 0:1, :]  # CLS token: (B, 1, 384)
        regs_tok = tokens[:, 1:1 + self.num_registers, :]  # Register tokens: (B, 4, 384)
        patch_tok = tokens[:, 1 + self.num_registers:, :]  # Patch tokens: (B, N, 384)
        
        # Reshape patch tokens to spatial grid
        g = int(self.img_size // self.patch_size)  # ViT-B/14 grid size
        patches2d = rearrange(patch_tok, "b (h w) d -> b h w d", h=g, w=g)
        
        # Combine CLS and register tokens to form cues
        cues = torch.cat([cls_tok, regs_tok], dim=1)  # (B, 5, 384)
        
        # Apply buddy pooling to get ROIs with straight-through estimator
        rois = torch.stack([self._buddy_pool(cues[:, i], patches2d, valid_masks)
                           for i in range(cues.size(1))], dim=1)
        
        # Combine cues and ROIs
        toks = torch.cat([cues, rois], dim=1)  # (B, 10, 384)
        
        # Normalize
        return F.normalize(toks, dim=-1)


#------// Dataset Class for Gallery Building //-----------------

class GalleryDataset(Dataset):
    """Dataset for efficient gallery building with DataLoader."""
    
    def __init__(self, file_list, inshop_root):
        self.file_list = file_list  # List of (file_path, item_id) tuples
        self.inshop_root = Path(inshop_root)
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
        file_path, item_id = self.file_list[idx]
        
        # Handle both relative and absolute paths
        if Path(file_path).is_absolute():
            full_path = Path(file_path)
        else:
            full_path = self.inshop_root / file_path
        
        try:
            img = Image.open(full_path).convert("RGB")
            img_tensor = self.transform(img)
            return img_tensor, item_id, str(full_path)
        except (FileNotFoundError, OSError) as e:
            print(f"Warning: Error loading {full_path}: {e}")
            # Return a dummy tensor
            dummy_tensor = torch.zeros(3, 224, 224)
            return dummy_tensor, item_id, str(full_path)


#------// Model Params //-----------------

model_img_size = 224


#------// Query Dataset Class for Batch Processing //-----------------

class QueryDataset(Dataset):
    """Dataset for batch processing of query images with padding/scaling."""
    
    def __init__(self, query_list, padding_scale=1.0, valid_threshold=0.5):
        self.query_list = query_list  # List of (query_path, item_id) tuples
        self.padding_scale = padding_scale
        self.valid_threshold = valid_threshold
    
    def __len__(self):
        return len(self.query_list)
    
    def __getitem__(self, idx):
        query_path, item_id = self.query_list[idx]
        
        # Apply padding/scaling to query
        img_tensor, valid_mask = self._apply_padding_and_mask(query_path, self.padding_scale)
        
        return img_tensor, valid_mask, item_id, query_path
    
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
        print(f"   - Training items: {config.get('train_items_count', 'Unknown')}")
    
    return model, checkpoint


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


def maxsim_scorer(q, g, chunk=2048):
    """ColBERT MaxSim scorer with chunking for memory efficiency."""
    sims_chunks = []
    for g_chunk in torch.split(g, chunk, dim=0):
        s = torch.einsum('btd,gkd->btgk', q, g_chunk)
        s = s.max(dim=-1).values
        s = s.sum(dim=1)
        sims_chunks.append(s)
    return torch.cat(sims_chunks, dim=1)


def _apply_padding_and_mask(image_path, padding_scale, target_size=224):
    """Apply padding to image and create valid mask - moved from QueryDataset for efficiency."""
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
        valid_mask = _create_valid_mask(padding_scale, target_size)
        
        return img_tensor, valid_mask
        
    except Exception as e:
        print(f"Warning: Error processing {image_path}: {e}")
        # Return dummy data
        dummy_tensor = torch.zeros(3, target_size, target_size)
        dummy_mask = torch.ones(16, 16, dtype=torch.bool)  # 224//14 = 16
        return dummy_tensor, dummy_mask

def _create_valid_mask(padding_scale, img_size=224, patch_size=14):
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

def evaluate_retrieval_full_dataset(model, checkpoint_dir, inshop_root, eval_batch_size, device, ks=[1, 10, 20, 30], valid_threshold=0.5, padding_scales=[0.6, 0.8, 1.0, 1.2, 1.4]):
    """Evaluate retrieval performance on In-Shop dataset using Query vs Gallery with multiple paddings combined by MAX."""
    print(f"\nEvaluating on In-Shop dataset with multiple padding scales (MAX combination)")
    print("="*50)
    
    # Load In-Shop dataset partitions
    train_paths, query_paths, gallery_paths = parse_inshop_partitions(inshop_root)
    
    if not gallery_paths:
        print(f"No gallery files found in In-Shop dataset")
        return None

    # Build gallery from gallery partition (not train!)
    print(f"Creating gallery dataset from {len(gallery_paths)} gallery items...")
    gallery_files = []
    for item_id, paths in gallery_paths.items():
        for path in paths:
            gallery_files.append((path, item_id))
    
    gallery_dataset = GalleryDataset(gallery_files, inshop_root)
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
    
    # Process gallery files efficiently with DataLoader
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
    print(f"Gallery items: {len(set(gallery_labels))} unique items")
    print(f"Gallery images: {len(gallery_items)} total images")
    
    if not query_paths:
        print(f"No query files found in In-Shop dataset")
        return None
    
    # Build query files from query partition
    query_files = []
    for item_id, paths in query_paths.items():
        for path in paths:
            query_files.append((path, item_id))
    
    # Combined evaluation with MAX across all scales using original DataLoader approach
    print(f"\nProcessing ALL padding scales with MAX combination: {padding_scales}")

    hits = {k: 0 for k in ks}
    aps = []
    total_queries = len(query_files)
    num_scales = len(padding_scales)

    print("Memory-optimized: Process batches immediately, avoid storing embeddings...")

    # Calculate how many images to process per batch (total batch size / num_scales)
    images_per_batch = eval_batch_size // num_scales
    total_model_batch_size = images_per_batch * num_scales  # Should equal eval_batch_size
    print(f"   Processing {images_per_batch} images × {num_scales} scales = {total_model_batch_size} total batch size")
    
    # Initialize storage for final similarities only (much smaller than embeddings!)
    query_similarities_all_scales = []  # Store final combined similarities only
    query_labels_list = []  # Store query labels
    
    model.eval()
    with torch.no_grad():
        # Process queries in batches
        for i in tqdm(range(0, len(query_files), images_per_batch), desc="Memory-optimized processing"):
            batch_files = query_files[i:i+images_per_batch]
            
            # Prepare all scales for this batch
            multi_scale_imgs = []
            multi_scale_masks = []
            batch_labels = []
            
            # For each image in batch (e.g., 25 images)
            for query_path, item_id in batch_files:
                # Create all scale versions of this image (e.g., 5 scales per image)
                for padding_scale in padding_scales:
                    img_tensor, valid_mask = _apply_padding_and_mask(query_path, padding_scale)
                    multi_scale_imgs.append(img_tensor)
                    multi_scale_masks.append(valid_mask)
                
                # Labels are same for all scales of this image
                batch_labels.append(item_id)
            
            if multi_scale_imgs:
                # Stack all scale versions: [img1_s1, img1_s2, ..., img1_s5, img2_s1, ..., img25_s5]
                # Total: 25 images × 5 scales = 125 tensors
                all_imgs = torch.stack(multi_scale_imgs).to(device)
                all_masks = torch.stack(multi_scale_masks).to(device)
                
                # Process ALL images in one forward pass
                all_embs = model(all_imgs, all_masks)  # (125, 10, 384)
                all_embs = F.normalize(all_embs, dim=-1)
                
                # Reshape back to separate scales: [25, 5, 10, 384]
                num_images = len(batch_files)
                all_embs = all_embs.view(num_images, num_scales, 10, 384)
                
                # IMMEDIATELY compute similarities for each query to avoid memory accumulation
                for img_idx in range(num_images):
                    query_label = batch_labels[img_idx]
                    query_sims_across_scales = []
                    
                    # Get similarities for this query across all scales
                    for scale_idx in range(num_scales):
                        query_emb = all_embs[img_idx:img_idx+1, scale_idx:scale_idx+1, :, :]  # (1, 1, 10, 384)
                        query_emb = query_emb.squeeze(1)  # (1, 10, 384)
                        sim = maxsim_scorer(query_emb, gallery_stack)  # (1, G)
                        query_sims_across_scales.append(sim.squeeze(0).cpu())  # (G,) on CPU
                    
                    # Stack similarities and take MAX across scales immediately
                    query_sim_stack = torch.stack(query_sims_across_scales)  # (num_scales, G)
                    combined_sim = query_sim_stack.max(dim=0).values  # (G,)
                    
                    # Store only the final combined similarity (much smaller!)
                    query_similarities_all_scales.append(combined_sim)
                    query_labels_list.append(query_label)
                
                # Free GPU memory immediately after processing batch
                del all_embs, all_imgs, all_masks
                torch.cuda.empty_cache()
        
        # Now compute final metrics from stored similarities (no more memory accumulation!)
        print("Computing final metrics from pre-computed similarity scores...")
        
        if not query_similarities_all_scales:
            print("No similarity scores computed")
            return None
        
        # Process final similarities to compute metrics
        for query_idx in tqdm(range(len(query_similarities_all_scales)), desc="Computing final metrics"):
            combined_sim = query_similarities_all_scales[query_idx]  # Already MAX-combined
            query_label = query_labels_list[query_idx]
            
            # Compute topk for recall metrics
            topk_indices = combined_sim.topk(max(ks), dim=0).indices
            
            # Recall@k
            for k in ks:
                if any(gallery_items[idx][0] == query_label for idx in topk_indices[:k]):
                    hits[k] += 1
            
            # Average Precision (AP)
            sorted_indices = torch.argsort(combined_sim, descending=True)
            
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
    total_processed_queries = len(query_similarities_all_scales)
    final_results = {}
    for k in ks:
        recall = hits[k] / total_processed_queries if total_processed_queries > 0 else 0.0
        final_results[f'recall@{k}'] = recall
    
    mAP = sum(aps) / len(aps) if aps else 0.0
    final_results['mAP'] = mAP
    final_results['total_queries'] = total_processed_queries
    
    # Print final summary
    print("\n" + "="*60)
    print("In-Shop Instance Retrieval Evaluation Results")
    print("   (Using Query images vs Gallery images with MAX across padding scales)")
    print("="*60)
    print(f"Strategy: MAX combination across scales {padding_scales}")
    print(f"Total queries processed: {total_processed_queries}")
    for k in ks:
        print(f"   Recall@{k}: {final_results[f'recall@{k}']:.4f} ({hits[k]}/{total_processed_queries})")
    print(f"   mAP: {mAP:.4f}")
    print("="*60)
    
    # Save detailed results
    results_file = checkpoint_dir / f"eval_results_inshop_max_combination_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
    with open(results_file, 'w') as f:
        f.write(f"In-Shop Instance Retrieval Evaluation Results (MAX Combination)\n")
        f.write(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write("="*60 + "\n")
        f.write(f"Model: Buddy Pooling + Straight-Through Estimator\n")
        f.write(f"Dataset: In-Shop Clothes Retrieval Benchmark\n")
        f.write(f"Task: INSTANCE-LEVEL retrieval (item_id-based)\n")
        f.write(f"Strategy: MAX combination across scales {padding_scales}\n")
        f.write(f"Total Gallery Items: {len(gallery_items)}\n")
        f.write(f"Total Query Items: {total_processed_queries}\n\n")
        
        f.write("Combined Results (MAX across all scales):\n")
        for k in ks:
            f.write(f"  Recall@{k}: {final_results[f'recall@{k}']:.4f}\n")
        f.write(f"  mAP: {mAP:.4f}\n")
    
    print(f"Results saved to: {results_file}")
    
    return final_results


@app.function(
    gpu="A100-80GB:1",  # Single A100 for evaluation
    timeout=8000  # 40 minutes timeout
)
def main(
    checkpoint_path: str = "/mnt/data/Checkpoints/aggressive_inshop_hardneg_estimator_20250822_222951",
    inshop_root: str = "/mnt/data",
    eval_batch_size: int = 125,  # Optimized: divisible by 5 scales
    valid_threshold: float = 0.9,  # Valid mask threshold for padding
    padding_scales = [0.1, 0.2, 0.4, 0.6, 0.8, 1.0],  # Multiple padding scales
):
    """
    Evaluate a trained In-Shop model on instance retrieval using MAX combination across multiple padding scales.
    
    Args:
        checkpoint_path: Path to model checkpoint (directory or .pth file)
        inshop_root: Path to In-Shop dataset root
        eval_batch_size: Total batch size for model (should be divisible by num_scales)
        valid_threshold: Threshold for valid mask computation
        padding_scales: List of padding scales to apply to queries (MAX combination used)
    """
    print(f"Starting In-Shop Instance Retrieval Model Evaluation with MAX Combination")
    print(f"   Checkpoint: {checkpoint_path}")
    print(f"   Dataset: In-Shop Clothes Retrieval Benchmark")
    print(f"   Task: INSTANCE-LEVEL retrieval (item_id-based)")
    print(f"   Padding scales: {padding_scales} (MAX combination)")
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
    
    # Evaluate on In-Shop dataset
    print(f"\n{'='*20} IN-SHOP EVALUATION with MULTIPLE PADDINGS {'='*20}")

    try:
        results = evaluate_retrieval_full_dataset(
            model, checkpoint_dir, inshop_root, eval_batch_size, device,
            ks=[1, 10, 20, 30], valid_threshold=valid_threshold, padding_scales=padding_scales
        )

        if results is None:
            print("Evaluation failed")
            return {"status": "failed", "error": "evaluation_failed"}

        print("Evaluation completed successfully!")

        return {
            "status": "success",
            "checkpoint_path": checkpoint_path,
            "dataset": "inshop_clothes_retrieval",
            "task": "instance_level_retrieval",
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
            checkpoint_path="/mnt/data/Checkpoints/aggressive_inshop_hardneg_estimator_20250101_120000",
            eval_batch_size=125,
            padding_scales=[0.6, 0.8, 1.0, 1.2, 1.4]
        )
        print("In-Shop Evaluation Result:", result) 
