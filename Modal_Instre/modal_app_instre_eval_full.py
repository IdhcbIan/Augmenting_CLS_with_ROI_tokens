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
    "INSTRE EVAL HARDNEGATIVE ESTIMATOR PAD AND SCALE",
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
    """TRAINABLE Multi-vector encoder with straight-through estimator (same as training script)."""
    
    def __init__(self):
        super().__init__()
        MODEL_NAME = "vit_base_patch14_reg4_dinov2.lvd142m"
        
        # Configuration matching original
        self.embed_dim = 768
        self.num_registers = 4
        self.img_size = model_img_size
        self.roi_side = 3
        self.patch_size = 14
        
        # Create the model - trainable with correct image size
        self.backbone = timm.create_model(MODEL_NAME, pretrained=True, num_classes=0, img_size=self.img_size)
        
        # Add a small projection layer to make it clearly trainable
        self.projection = nn.Linear(self.embed_dim, self.embed_dim)
        
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
    
    def __init__(self, file_list, instre_root):
        self.file_list = file_list  # List of (file_path, class_id) tuples
        self.instre_root = Path(instre_root)
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
        file_path, class_id = self.file_list[idx]
        
        # Handle both relative and absolute paths
        if Path(file_path).is_absolute():
            full_path = Path(file_path)
        else:
            full_path = self.instre_root / file_path
        
        try:
            img = Image.open(full_path).convert("RGB")
            img_tensor = self.transform(img)
            return img_tensor, class_id, str(full_path)
        except (FileNotFoundError, OSError) as e:
            print(f"  Error loading {full_path}: {e}")
            # Return a dummy tensor
            dummy_tensor = torch.zeros(3, 224, 224)
            return dummy_tensor, class_id, str(full_path)


#------// Model Params //-----------------

model_img_size = 224


#------// Query Dataset Class for Batch Processing //-----------------

class QueryDataset(Dataset):
    """Dataset for batch processing of query images with padding/scaling."""
    
    def __init__(self, query_list, padding_scale=1.0, valid_threshold=0.5):
        self.query_list = query_list  # List of (query_path, class_id) tuples
        self.padding_scale = padding_scale
        self.valid_threshold = valid_threshold
    
    def __len__(self):
        return len(self.query_list)
    
    def __getitem__(self, idx):
        query_path, class_id = self.query_list[idx]
        
        # Apply padding/scaling to query
        img_tensor, valid_mask = self._apply_padding_and_mask(query_path, self.padding_scale)
        
        return img_tensor, valid_mask, class_id, query_path
    
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
            print(f"  Error processing {image_path}: {e}")
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
    
    print(f" Loading checkpoint from: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    # Create model and load weights
    model = TrainableMultiVectorEncoder()
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    model.eval()
    
    print(f" Model loaded successfully!")
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


def get_equivalent_classes(class_name, cls_map):
    """
    Get all equivalent class names for a given class.
    
    INSTRE-M classes 200-249 contain objects from pairs of classes:
    - INSTRE-M class 200 contains objects from classes 0 and 1
    - INSTRE-M class 201 contains objects from classes 2 and 3  
    - INSTRE-M class 202 contains objects from classes 4 and 5
    - ...
    - INSTRE-M class 249 contains objects from classes 98 and 99
    
    For retrieval:
    - Query class 0 matches gallery classes: 0 and 200
    - Query class 1 matches gallery classes: 1 and 200
    - Query class 2 matches gallery classes: 2 and 201
    - etc.
    
    Args:
        class_name: The class name (e.g., "INSTRE-S1_01a_canada_book")
        cls_map: Dictionary mapping class IDs to class names
    """
    # Reverse lookup: find the class ID for this class name
    class_id = None
    for cid, cname in cls_map.items():
        if cname == class_name:
            class_id = cid
            break
    
    if class_id is None:
        return [class_name]  # No mapping found, return original
    
    equivalent_ids = []
    if 0 <= class_id <= 99:
        # Classes 0-99: match original class + corresponding INSTRE-M class
        instre_m_class = 200 + (class_id // 2)
        equivalent_ids = [class_id, instre_m_class]
    elif 200 <= class_id <= 249:
        # INSTRE-M classes: match both constituent classes
        base = (class_id - 200) * 2
        equivalent_ids = [base, base + 1, class_id]
    else:
        # Other classes (100-199) don't have equivalents in this mapping
        equivalent_ids = [class_id]
    
    # Convert IDs back to class names
    equivalent_names = []
    for eid in equivalent_ids:
        if eid in cls_map:
            equivalent_names.append(cls_map[eid])
    
    return equivalent_names if equivalent_names else [class_name]


def load_full_instre_files(instre_root):
    """Load the FULL INSTRE train and cropped query files (no threshold filtering)."""
    instre_root = Path(instre_root)
    
    print(f" Loading FULL INSTRE dataset (no threshold filtering):")
    
    # Find Queries_Cropped directory
    possible_cropped_paths = [
        instre_root / "Queries_Cropped",
        Path("/mnt/data/instre/Queries_Cropped"),
        Path("/mnt/data/instre_converted/Queries_Cropped")
    ]
    
    queries_cropped_root = None
    for path in possible_cropped_paths:
        if path.exists():
            queries_cropped_root = path
            print(f" Found Queries_Cropped at: {queries_cropped_root}")
            break
    
    if queries_cropped_root is None:
        raise FileNotFoundError("Queries_Cropped directory not found in any expected location")
    
    # Load FULL dataset by parsing INSTRE metadata files (same as training script)
    train_files = []
    query_files = []
    
    # Parse INSTRE files
    cls_map = {}
    for line in (instre_root / "classes.txt").read_text().splitlines():
        cid, cname = line.split()
        cls_map[int(cid)] = cname

    img_to_cid = {}
    for line in (instre_root / "image_class_labels.txt").read_text().splitlines():
        iid, cid = line.split()
        img_to_cid[int(iid)] = int(cid)

    img_map = {}
    for line in (instre_root / "images.txt").read_text().splitlines():
        iid, rel = line.split()
        full_path = instre_root / rel
        if not rel.startswith('._') and full_path.exists():
            img_map[int(iid)] = str(full_path)

    train_ids = set()
    for line in (instre_root / "train_test_split.txt").read_text().splitlines():
        iid, flag = line.split()
        if int(flag):
            train_ids.add(int(iid))

    # Create directory to class mapping for cropped queries
    dir_to_class = {}
    for iid, path in img_map.items():
        cid = img_to_cid[iid]
        class_name = cls_map[cid]
        directory = Path(path).parent.name
        dir_to_class[directory] = class_name

    # Build train files from FULL dataset
    for iid, path in img_map.items():
        if iid in train_ids:
            cid = img_to_cid[iid]
            class_name = cls_map[cid]
            train_files.append((path, class_name))
    
    # Build query files from Queries_Cropped directory
    total_directories = 0
    found_classes = []
    for class_dir in sorted(queries_cropped_root.iterdir()):
        if not class_dir.is_dir():
            continue

        total_directories += 1
        dir_name = class_dir.name  # e.g. "12"
        found_classes.append(dir_name)
        
        # Handle class naming based on file type
        cropped_files = list(class_dir.glob("*.jpg"))
        for crop_file in cropped_files:
            # For crop files, convert directory number to actual class name
            # For regular files, use original INSTRE mapping
            if "_crop" in crop_file.name:
                try:
                    # Convert directory name (e.g., "0") to class name (e.g., "INSTRE-S1_01a_canada_book")
                    class_id = int(dir_name)
                    class_name = cls_map.get(class_id)
                    if class_name is None:
                        continue  # Skip if class ID not found in cls_map
                except ValueError:
                    continue  # Skip if directory name is not a number
            else:
                class_name = dir_to_class.get(dir_name)  # Use original mapping for regular files
                if class_name is None:
                    continue  # Skip if no mapping found
            
            query_files.append((str(crop_file), class_name))
    
    print(f" Debug: Found {total_directories} query directories in Queries_Cropped")
    print(f" Debug: Class range: {min(found_classes) if found_classes else 'N/A'} to {max(found_classes) if found_classes else 'N/A'}")
    
    # Check for missing classes in range 0-199
    numeric_classes = []
    for cls in found_classes:
        try:
            numeric_classes.append(int(cls))
        except ValueError:
            pass
    
    if numeric_classes:
        missing_classes = []
        for i in range(200):  # Expected classes 0-199
            if i not in numeric_classes:
                missing_classes.append(i)
        
        if missing_classes:
            print(f"  Missing classes in range 0-199: {len(missing_classes)} classes missing")
            if len(missing_classes) <= 20:
                print(f"    Missing: {missing_classes}")
            else:
                print(f"    First 20 missing: {missing_classes[:20]}...")
    
    print(f" Successfully loaded FULL INSTRE dataset:")
    print(f"   Train files: {len(train_files)}")
    print(f"   Cropped query files: {len(query_files)} (Expected: 1250 for complete dataset)")
    
    # Debug: Check unique classes in train and query files
    train_classes = set(class_id for _, class_id in train_files)
    query_classes = set(class_id for _, class_id in query_files)
    print(f"   Train classes: {len(train_classes)} unique - {sorted(list(train_classes))[:10]}...")
    print(f"   Query classes: {len(query_classes)} unique - {sorted(list(query_classes))[:10]}...")
    
    # Debug: Show some examples of query files and their labels
    print(f"\n Sample query files and labels:")
    for i, (path, label) in enumerate(query_files[:10]):
        filename = Path(path).name
        print(f"   {filename} → class '{label}'")
    
    # Debug: Check class overlap
    overlap = train_classes.intersection(query_classes)
    print(f"\n Class overlap: {len(overlap)} classes appear in both train and query")
    print(f"   Example overlapping classes: {sorted(list(overlap))[:10]}...")
    
    return train_files, query_files, cls_map


def maxsim_scorer(q, g, chunk=2048):
    """ColBERT MaxSim scorer with chunking for memory efficiency."""
    sims_chunks = []
    for g_chunk in torch.split(g, chunk, dim=0):
        s = torch.einsum('btd,gkd->btgk', q, g_chunk)
        s = s.max(dim=-1).values
        s = s.sum(dim=1)
        sims_chunks.append(s)
    return torch.cat(sims_chunks, dim=1)


def evaluate_retrieval_full_dataset(model, checkpoint_dir, instre_root, eval_batch_size, device, ks=[1, 2, 4, 8], valid_threshold=0.5, padding_scales=[0.6, 0.8, 1.0, 1.2, 1.4]):
    """Evaluate retrieval performance on FULL INSTRE dataset using CROPPED queries with MULTIPLE PADDINGS combined by MAX."""
    print(f"\n Evaluating on FULL INSTRE dataset with multiple padding scales (MAX combination)")
    print("="*50)
    
    # Force dynamic gallery building for full dataset evaluation
    print("  Building gallery dynamically from FULL train files...")
    
    # Load FULL INSTRE files
    train_files, query_files, cls_map = load_full_instre_files(instre_root)
    
    if not train_files:
        print(f" No train files found in FULL INSTRE dataset")
        return None

    # Build gallery from train files using DataLoader
    print(f" Creating gallery dataset from {len(train_files)} train files...")
    gallery_dataset = GalleryDataset(train_files, instre_root)
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
            embs = model(batch_imgs)  # (B, 10, 768)
            
            # Store embeddings and metadata
            for emb, label, path in zip(embs, batch_labels, batch_paths):
                gallery_embeddings.append(emb.cpu())
                gallery_labels.append(label)
                gallery_items.append((label, path))
    
    if not gallery_embeddings:
        print(" No valid gallery embeddings created")
        return None
    
    gallery_stack = torch.stack(gallery_embeddings).to(device)  # (G, 10, 768)
    gallery_stack = F.normalize(gallery_stack, dim=-1)
    
    print(f" Gallery built: {gallery_stack.shape}")
    print(f" Gallery classes: {len(set(gallery_labels))} unique classes")
    print(f" Gallery items: {len(gallery_items)} total items")
    
    if not query_files:
        print(f" No query files found in FULL INSTRE dataset")
        return None
    
        # Combined evaluation with MAX across all scales using original DataLoader approach
    print(f"\n Processing ALL padding scales with MAX combination: {padding_scales}")
    
    hits = {k: 0 for k in ks}
    aps = []
    total_queries = len(query_files)
    num_scales = len(padding_scales)
    
    # Process each scale using the original DataLoader approach and collect all embeddings
    all_scale_embeddings = []  # List of (query_embs, query_labels) for each scale
    
    model.eval()
    with torch.no_grad():
        for scale_idx, padding_scale in enumerate(padding_scales):
            print(f" Processing scale {padding_scale} ({scale_idx+1}/{num_scales})")
            
            # Create query dataset for this padding scale (original approach)
            query_dataset = QueryDataset(query_files, padding_scale, valid_threshold)
            query_dataloader = DataLoader(
                query_dataset, 
                batch_size=eval_batch_size, 
                shuffle=False, 
                num_workers=2, 
                pin_memory=True
            )
            
            scale_embeddings = []
            scale_labels = []
            
            # Process queries for this scale using original DataLoader approach
            for batch_imgs, batch_masks, batch_labels, batch_paths in tqdm(query_dataloader, desc=f"Scale {padding_scale}"):
                # Move to device
                batch_imgs = batch_imgs.to(device)
                batch_masks = batch_masks.to(device)
                
                # Get query embeddings
                query_embs = model(batch_imgs, batch_masks)  # (B, 10, 768)
                query_embs = F.normalize(query_embs, dim=-1)
                
                # Store embeddings and labels for this scale
                scale_embeddings.append(query_embs.cpu())
                scale_labels.extend(batch_labels)
            
            # Concatenate all embeddings for this scale
            if scale_embeddings:
                scale_embs_tensor = torch.cat(scale_embeddings, dim=0)  # (total_queries, 10, 768)
                all_scale_embeddings.append((scale_embs_tensor, scale_labels))
        
        # Now combine similarities using MAX across all scales
        print(" Combining similarities using MAX across all scales...")
        
        if not all_scale_embeddings or len(all_scale_embeddings) != num_scales:
            print(" Failed to collect embeddings for all scales")
            return None
        
        # Process queries and combine similarities using MAX
        for query_idx in tqdm(range(total_queries), desc="Computing MAX similarities"):
            # Collect similarities for this query across all scales
            query_sims_across_scales = []
            query_label = all_scale_embeddings[0][1][query_idx]  # Get label from first scale
            
            for scale_embs, scale_labels in all_scale_embeddings:
                if query_idx < len(scale_embs):
                    query_emb = scale_embs[query_idx:query_idx+1].to(device)  # (1, 10, 768)
                    sim = maxsim_scorer(query_emb, gallery_stack)  # (1, G)
                    query_sims_across_scales.append(sim.squeeze(0))  # (G,)
            
            if len(query_sims_across_scales) == num_scales:
                # Stack and take MAX across scales
                query_sim_stack = torch.stack(query_sims_across_scales)  # (num_scales, G)
                combined_sim = query_sim_stack.max(dim=0).values  # (G,)
                
                # Compute topk for recall metrics
                topk_indices = combined_sim.topk(max(ks), dim=0).indices.cpu()
                
                # Recall@k - consider equivalent classes
                query_equivalent_classes = get_equivalent_classes(query_label, cls_map)
                
                # Debug: Print first few queries to see what's happening
                if query_idx < 3:
                    print(f"\n Debug Query {query_idx}:")
                    print(f"   Query label: '{query_label}'")
                    print(f"   Equivalent classes: {query_equivalent_classes}")
                    print(f"   Top-5 gallery labels: {[gallery_items[idx][0] for idx in topk_indices[:5]]}")
                
                for k in ks:
                    if any(gallery_items[idx][0] in query_equivalent_classes for idx in topk_indices[:k]):
                        hits[k] += 1
                
                # Average Precision (AP)
                sorted_indices = torch.argsort(combined_sim, descending=True).cpu()
                
                # Compute AP with full gallery depth - consider equivalent classes
                MAP_DEPTH = len(gallery_items)
                relevances = [1 if gallery_items[idx][0] in query_equivalent_classes else 0 
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
    print(" FULL INSTRE RETRIEVAL EVALUATION RESULTS (Hardnegative Estimator)")
    print("   (Using CROPPED queries vs full training images with MAX across padding scales)")
    print("="*60)
    print(f" Strategy: MAX combination across scales {padding_scales}")
    print(f" Total queries processed: {total_queries}")
    for k in ks:
        print(f"   Recall@{k}: {final_results[f'recall@{k}']:.4f} ({hits[k]}/{total_queries})")
    print(f"   mAP: {mAP:.4f}")
    print("="*60)
    
    # Save detailed results
    results_file = checkpoint_dir / f"eval_results_full_dataset_max_combination_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
    with open(results_file, 'w') as f:
        f.write(f"INSTRE Full Dataset Evaluation Results - Hardnegative Estimator (MAX Combination)\n")
        f.write(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write("="*60 + "\n")
        f.write(f"Model: Aggressive Hard Negative Mining + Straight-Through Estimator\n")
        f.write(f"Dataset: Full INSTRE (no threshold filtering)\n")
        f.write(f"Strategy: MAX combination across scales {padding_scales}\n")
        f.write(f"Total Gallery Items: {len(gallery_items)}\n")
        f.write(f"Total Query Items: {total_queries}\n\n")
        
        f.write("Combined Results (MAX across all scales):\n")
        for k in ks:
            f.write(f"  Recall@{k}: {final_results[f'recall@{k}']:.4f}\n")
        f.write(f"  mAP: {mAP:.4f}\n")
    
    print(f" Results saved to: {results_file}")
    
    return final_results


@app.function(
    gpu="A100-80GB:1",  # Single A100 for evaluation
    timeout=2400  # 40 minutes timeout
)
def main(
    checkpoint_path: str = "/mnt/data/Checkpoints/model_20250712_143943",
    instre_root: str = "/mnt/data/instre_converted",
    eval_batch_size: int = 64,  # Smaller batch size due to multiple paddings
    valid_threshold: float = 0.5,  # Valid mask threshold for padding
    padding_scales = [0.1, 0.2,0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0],  # Multiple padding scales
):
    """
    Evaluate a trained INSTRE Hardnegative Estimator model on FULL dataset using MAX combination across multiple padding scales.
    
    Args:
        checkpoint_path: Path to model checkpoint (directory or .pth file)
        instre_root: Path to instre_converted directory
        eval_batch_size: Batch size for evaluation (smaller due to multiple paddings)
        valid_threshold: Threshold for valid mask computation
        padding_scales: List of padding scales to apply to queries (MAX combination used)
    """
    print(f" Starting INSTRE Hardnegative Estimator Model Evaluation with MAX Combination")
    print(f"   Checkpoint: {checkpoint_path}")
    print(f"   Dataset: FULL INSTRE (no threshold filtering)")
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
    
    # Evaluate on full dataset
    print(f"\n{'='*20} FULL DATASET with MULTIPLE PADDINGS {'='*20}")
    
    try:
        results = evaluate_retrieval_full_dataset(
            model, checkpoint_dir, instre_root, eval_batch_size, device,
            ks=[1, 2, 4, 8], valid_threshold=valid_threshold, padding_scales=padding_scales
        )
        
        if results is None:
            print(" Evaluation failed")
            return {"status": "failed", "error": "evaluation_failed"}
        
        print(" Evaluation completed successfully!")
        
        return {
            "status": "success",
            "checkpoint_path": checkpoint_path,
            "dataset": "full_instre",
            "padding_scales": padding_scales,
            "combination_strategy": "max",
            "results": results,
            "timestamp": datetime.now().strftime('%Y%m%d_%H%M%S')
        }
        
    except Exception as e:
        print(f" Error during evaluation: {e}")
        import traceback
        traceback.print_exc()
        return {"status": "error", "error": str(e)}


if __name__ == "__main__":
    # Example usage
    with app.run():
        result = main.remote(
            checkpoint_path="/mnt/data/Checkpoints/aggressive_hardneg_full_estimator_20250728_165641",
            eval_batch_size=64,
            padding_scales=[0.6, 0.8, 1.0, 1.2, 1.4]
        )
        print("Evaluation Result:", result) 
