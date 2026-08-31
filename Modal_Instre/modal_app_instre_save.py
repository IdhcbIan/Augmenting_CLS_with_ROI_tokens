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
    "INSTRE V1",
    image=image,
    volumes={"/mnt/data": modal.Volume.from_name("instre_converted")}
)




class TrainableMultiVectorEncoder(nn.Module):
    """TRAINABLE Multi-vector encoder."""
    
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
        """Original buddy pooling implementation."""
        B, H, W, d = patches2d.shape
        flat = rearrange(patches2d, "b h w d -> b (h w) d")
        sim = torch.matmul(cue.unsqueeze(1), flat.transpose(1, 2)).squeeze(1)
        idx = sim.argmax(dim=-1)
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
        """Clean forward pass - no checkpointing."""
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
        
        # Apply buddy pooling to get ROIs
        rois = torch.stack([self._buddy_pool(cues[:, i], patches2d)
                           for i in range(cues.size(1))], dim=1)
        
        # Combine cues and ROIs
        toks = torch.cat([cues, rois], dim=1)  # (B, 10, D)
        
        # Normalize
        return F.normalize(toks, dim=-1)


def load_model_checkpoint(checkpoint_path, device='cuda:0'):
    """
    Load a saved model checkpoint.
    
    Args:
        checkpoint_path: Path to the checkpoint.pth file or directory containing it
        device: Device to load the model on
    
    Returns:
        tuple: (model, checkpoint_data) where checkpoint_data contains training info
    """
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
    print(f"   - Recall@1: {checkpoint.get('recall_at_1', 'Unknown')}")
    print(f"   - mAP: {checkpoint.get('mAP', 'Unknown')}")
    
    return model, checkpoint


def load_model_weights_only(weights_path, device='cuda:0'):
    """
    Load only the model weights (faster for inference).
    
    Args:
        weights_path: Path to the model_weights.pth file or directory containing it
        device: Device to load the model on
    
    Returns:
        model: Loaded model ready for inference
    """
    if Path(weights_path).is_dir():
        weights_path = Path(weights_path) / "model_weights.pth"
    
    print(f" Loading model weights from: {weights_path}")
    
    # Create model and load weights
    model = TrainableMultiVectorEncoder()
    model.load_state_dict(torch.load(weights_path, map_location=device))
    model = model.to(device)
    model.eval()
    
    print(f" Model weights loaded successfully!")
    return model


def load_gallery_data(gallery_path, device='cuda:0'):
    """
    Load saved gallery data.
    
    Args:
        gallery_path: Path to the gallery.pth file or directory containing it
        device: Device to load the gallery on
    
    Returns:
        gallery_data: Dictionary containing gallery embeddings, labels, and metadata
    """
    if Path(gallery_path).is_dir():
        gallery_path = Path(gallery_path) / "gallery.pth"
    
    print(f" Loading gallery data from: {gallery_path}")
    gallery_data = torch.load(gallery_path, map_location=device)
    
    print(f" Gallery data loaded successfully!")
    print(f"   - Gallery embeddings: {gallery_data['embedding_shape']}")
    print(f"   - Gallery items: {gallery_data['num_items']}")
    print(f"   - Gallery classes: {len(gallery_data['classes'])}")
    
    return gallery_data





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



#------// Model Params //-----------------


model_img_size = 224

def _load_image(path):
    """Load a PIL image and preprocess it to tensor."""
    preprocess = transforms.Compose([
        transforms.Resize(model_img_size),
        transforms.CenterCrop(model_img_size),  # Match img_size
        #transforms.RandomResizedCrop(model_img_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                           std=[0.229, 0.224, 0.225]),
    ])
    try:
        img = Image.open(path).convert("RGB")
        return preprocess(img)  # Returns tensor
    except (FileNotFoundError, OSError) as e:
        print(f"  Skipping file {path}: {e}")
        return None




#------// DinoV2 Params //-----------------


class INSTRETripletDataset(Dataset):
    """Efficient Dataset for INSTRE triplet sampling with caching."""
    
    def __init__(self, class_to_paths, dir_to_class, steps_per_epoch=1000):
        self.class_to_paths = class_to_paths
        self.dir_to_class = dir_to_class
        self.classes = list(class_to_paths.keys())
        self.steps_per_epoch = steps_per_epoch
        
        # Pre-sample triplets for the epoch to reduce random access
        self._generate_epoch_triplets()
        
    def _generate_epoch_triplets(self):
        """Pre-generate triplets for the entire epoch."""
        self.triplets = []
        for _ in range(self.steps_per_epoch):
            cls_pos = random.choice(self.classes)
            a = random.choice(self.class_to_paths[cls_pos])
            p = random.choice(self.class_to_paths[cls_pos])
            neg_cls = random.choice([c for c in self.classes if c != cls_pos])
            n = random.choice(self.class_to_paths[neg_cls])
            self.triplets.append((a, p, n))
            
            # Debug: Verify class assignments using directory-based mapping
            if _ < 3:  # Only check first 3 triplets for debugging
                a_filename = Path(a).name
                p_filename = Path(p).name
                n_filename = Path(n).name
                a_dir = Path(a).parent.name
                p_dir = Path(p).parent.name
                n_dir = Path(n).parent.name
                
                a_class = self.dir_to_class.get(a_dir, "UNKNOWN")
                p_class = self.dir_to_class.get(p_dir, "UNKNOWN")
                n_class = self.dir_to_class.get(n_dir, "UNKNOWN")
                
                print(f" Triplet {_+1}: A({a_filename} from dir {a_dir}->{a_class}) P({p_filename} from dir {p_dir}->{p_class}) N({n_filename} from dir {n_dir}->{n_class})")
                print(f"   Expected: A&P should be same class ({cls_pos}), N should be different ({neg_cls})")

    def __len__(self):
        return self.steps_per_epoch
    
    def __getitem__(self, idx):
        a_path, p_path, n_path = self.triplets[idx]
        
        # Load images with error handling
        anchor = _load_image(a_path)
        positive = _load_image(p_path) 
        negative = _load_image(n_path)
        
        # If any image failed to load, try different triplet
        if anchor is None or positive is None or negative is None:
            # Generate a new triplet on the fly
            return self.__getitem__((idx + 1) % len(self))
        
        return anchor, positive, negative





#------// Modal Params //-----------------

@app.function(
    gpu="A100-80GB:2",  # 2 A100-80GB GPUs
    timeout=10800  # 3 hour timeout
)


def main(
    instre_root: str = "/mnt/data/instre_converted",
    steps: int = 1000,
    batch_size: int = 256,   
    report_interval: int = 1,
    eval_batch_size: int = 256,  
    lr: float = 1e-5  # Lower learning rate for fine-tuning
):
    """
    Train INSTRE triplet model on Modal with multiple A100 GPUs.
    Simple FP32 training matching original aug_cls_repo approach.
    """
    # Setup multi-GPU environment
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
    
    instre_root_path = Path(instre_root)

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

        """
        # OPTIMIZATION: Limit large classes to prevent sampling bias and improve speed
        #MAX_IMAGES_PER_CLASS = 200  # Limit very large classes for balanced training
        MAX_IMAGES_PER_CLASS = 20000  # Limit very large classes for balanced training
        for class_name in train_paths:
            if len(train_paths[class_name]) > MAX_IMAGES_PER_CLASS:
                print(f"     Limiting {class_name} from {len(train_paths[class_name])} to {MAX_IMAGES_PER_CLASS} images")
                train_paths[class_name] = train_paths[class_name][:MAX_IMAGES_PER_CLASS]
        """

        # OPTIMIZATION: Convert all paths to strings to reduce Path object overhead
        print(" Converting paths to strings for faster access...")
        train_paths = {k: [str(p) for p in v] for k, v in train_paths.items()}
        test_paths = {k: [str(p) for p in v] for k, v in test_paths.items()}
        
        return train_paths, test_paths, filename_to_class, path_to_class, dir_to_class












    def evaluate_retrieval_recalls(train_paths, test_paths, model, device, ks, eval_batch_size, num_gpus, filename_to_class, path_to_class, dir_to_class):
        """
        Evaluate retrieval performance using CROPPED queries vs training gallery.
        This is the proper INSTRE evaluation protocol that focuses on object instances.
        Returns: (recall_at_1, mAP, gallery_data)
        """
        model.eval()
        
        # Calculate effective evaluation batch size for multi-GPU
        effective_eval_batch_size = eval_batch_size * num_gpus
        print(f" Evaluation using effective batch size: {effective_eval_batch_size} (base: {eval_batch_size} × {num_gpus} GPUs)")
        
        classes = sorted(train_paths.keys())
        cls2idx = {c: i for i, c in enumerate(classes)}
        gallery_embs, gallery_labels = [], []
        
        # Build gallery from training set with batched processing
        with torch.no_grad():
            # Take a representative sample from each class for the gallery
            gallery_items = []
            for c in classes:
                # Take up to 10 images per class for gallery (balanced representation)
                #sample_size = min(10, len(train_paths[c]))
                sample_size = len(train_paths[c])
                print(f" Class {c} has {sample_size} images")

                selected_paths = train_paths[c][:sample_size]
                for p in selected_paths:
                    gallery_items.append((c, p))
            
            print(f" Building gallery with {len(gallery_items)} images...")
            
            # Debug: Show some example gallery assignments
            print(" Example gallery assignments:")
            for i, (class_name, path) in enumerate(gallery_items[:5]):
                dir_name = Path(path).parent.name
                filename = Path(path).name
                #print(f"   Gallery {i+1}: {filename} from dir {dir_name} -> class {class_name}")

            print(f" Building gallery with {len(gallery_items)} images...")
            for i in trange(0, len(gallery_items), effective_eval_batch_size, desc="Building gallery", unit="batch"):
                batch = gallery_items[i:i+effective_eval_batch_size]
                imgs = []
                batch_labels = []
                
                for c, p in batch:
                    img = _load_image(p)  # Already returns tensor
                    if img is not None:  # Only add if image loaded successfully
                        imgs.append(img)
                        batch_labels.append(c)  # Use class name directly
                
                if imgs:  # Only process if we have images
                    imgs = torch.stack(imgs).to(device)
                    embs = model(imgs)
                    
                    # ---- keep FULL 10-token descriptor in the gallery ----
                    for j, emb in enumerate(embs):               # emb: (10, 768)
                        gallery_embs.append(emb.cpu())           # no [:0] slicing!
                        gallery_labels.append(batch_labels[j])
        
        gallery_stack = torch.stack(gallery_embs)    # (G, 10, 768)
        gallery_stack = F.normalize(gallery_stack, dim=-1)
        
        # Store gallery_items for later use in evaluation
        global_gallery_items = gallery_items

        # ===== UPDATED: Use CROPPED queries and add to filename mapping =====
        print(" Loading CROPPED queries for proper INSTRE evaluation...")
        query_items = []
        
        # Try multiple possible locations for Queries_Cropped without affecting training
        possible_query_paths = [
            #instre_root_path / "Queries_Cropped",
            #instre_root_path / "instre_converted" / "Queries_Cropped",
            Path("/mnt/data/instre/Queries_Cropped")
            #Path("/mnt/data/instre_converted/instre_converted/Queries_Cropped")
        ]
        
        cropped_queries_root = None
        for path in possible_query_paths:
            if path.exists():
                cropped_queries_root = path
                print(10 *"=")
                time.sleep(2)
                print(f" Found Queries_Cropped at: {cropped_queries_root}")
                print(10 *"=")
                break
        
        # Fallback: attempt a recursive search if still not found
        if cropped_queries_root is None:
            for dirpath in instre_root_path.rglob("Queries_Cropped"):
                if dirpath.is_dir():
                    cropped_queries_root = dirpath
                    print(f" Found Queries_Cropped via recursive search at: {cropped_queries_root}")
                    break
        
        if cropped_queries_root is None:
            print(f"  Queries_Cropped directory not found in any of these locations:")
            for path in possible_query_paths:
                print(f"   - {path}")
        
        if cropped_queries_root and cropped_queries_root.exists():
            print(f" Using Queries_Cropped directory at {cropped_queries_root}")
            
            # Load cropped queries from each class directory
            for class_dir in sorted(cropped_queries_root.iterdir()):
                if not class_dir.is_dir():
                    continue

                dir_name = class_dir.name  # e.g. "12"

                # Resolve to the canonical class name used in the rest of the code.
                # If the directory isn't present in `dir_to_class` we skip it.
                class_name = dir_to_class.get(dir_name)
                if class_name is None or class_name not in train_paths:
                    continue  # Skip unknown / filtered classes

                # Collect (up to) a few query crops for the class
                cropped_files = list(class_dir.glob("*.jpg"))
                sample_size = min(5, len(cropped_files))
                for crop_file in cropped_files[:sample_size]:
                    crop_filename = crop_file.name
                    crop_path = str(crop_file)
                    
                    query_items.append((class_name, crop_path))
                    print(f"    Added cropped query: {crop_filename} from dir {dir_name} -> {class_name}")
            
            print(f" Found {len(query_items)} cropped queries across {len(classes)} classes")
        
        print(" Example query assignments:")
        for i, (class_name, path) in enumerate(query_items[:5]):
            dir_name = Path(path).parent.name
            filename = Path(path).name
            print(f"   Query {i+1}: {filename} from dir {dir_name} -> class {class_name}")
        
        print(f" Evaluating {len(query_items)} queries...")
        
        total = len(query_items)
        hits = {k: 0 for k in ks}
        aps = []  # Track Average Precision for every query

        # Process queries with effective batch size
        for i in trange(0, total, effective_eval_batch_size, desc="Evaluating queries", unit="batch"):
            batch = query_items[i:i+effective_eval_batch_size]
            valid_items = []
            imgs = []
            for class_name, p in batch:
                img = _load_image(p)  # Already returns tensor
                if img is not None:  # Only add if image loaded successfully
                    imgs.append(img)
                    valid_items.append(class_name)
            
            if not imgs:  # Skip batch if no valid images
                continue
                
            imgs = torch.stack(imgs).to(device)

            with torch.no_grad():
                embs = model(imgs)
                
                # ---- obtain the 10-token query descriptor ----
                queries = F.normalize(embs.cpu(), dim=-1)    # (B, 10, 768)

                # ---- ColBERT MaxSim scorer, chunked so it fits in RAM ----
                def maxsim(q, g, chunk=2048):
                    """
                    q : (B, 10, D)   g : (G, 10, D)
                    returns (B, G)   – ColBERT MaxSim(q, g)
                    """
                    sims_chunks = []
                    for g_chunk in torch.split(g, chunk, dim=0):          # (g_c, 10, D)
                        # (B,10,D) • (g_c,10,D) -> (B,10,g_c,10)
                        s = torch.einsum('btd,gkd->btgk', q, g_chunk)     # pairwise dot
                        s = s.max(dim=-1).values                          # max over *gallery* token k
                        s = s.sum(dim=1)                                  # sum over query tokens t
                        sims_chunks.append(s)                             # (B, g_c)
                    return torch.cat(sims_chunks, dim=1)                  # (B, G)

                # run scorer
                sims = maxsim(queries.to(device), gallery_stack.to(device))   # (B, G)
                topk = sims.topk(max(ks), dim=1).indices.cpu().tolist()

            # ----- Compute Recall and Average-Precision for this batch -----
            for qi, row_topk in enumerate(topk):
                query_class = valid_items[qi]

                # ----- Recall@k -----
                for k in ks:
                    if any(global_gallery_items[idx][0] == query_class for idx in row_topk[:k]):
                        hits[k] += 1

                # ----- Average Precision -----
                sims_row = sims[qi]
                sorted_idx = torch.argsort(sims_row, descending=True).cpu().tolist()
                
                # FIXED: Limit depth for mAP calculation (typically top-100 or top-1000)
                #MAP_DEPTH = 100  # or 1000 for more comprehensive evaluation
                #MAP_DEPTH = 1000  # or 1000 for more comprehensive evaluation
                #MAP_DEPTH = len(sorted_idx)  # Use all gallery images 
                MAP_DEPTH = 999999  # Large Number!!
                sorted_idx_limited = sorted_idx[:MAP_DEPTH]
                relevances = [1 if global_gallery_items[idx][0] == query_class else 0 for idx in sorted_idx_limited]

                num_rel, ap_sum = 0, 0.0
                for rank, rel in enumerate(relevances, start=1):
                    if rel:
                        num_rel += 1
                        ap_sum += num_rel / rank
                aps.append(ap_sum / num_rel if num_rel > 0 else 0.0)

        # Compute Mean Average Precision
        mAP = sum(aps) / len(aps) if aps else 0.0

        # Report results
        print("\n" + "="*40)
        print(" INSTRE RETRIEVAL EVALUATION RESULTS")
        print("   (Using CROPPED queries vs full training images)")
        print("    Class verification: Using filename-to-class mapping")
        print("    Method: Check if retrieved vectors are from same class (directory number)")
        print("="*40)
        for k in ks:
            recall = hits[k] / total
            print(f"Recall@{k}: {recall:.4f} ({hits[k]}/{total})")
        print(f"mAP: {mAP:.4f}")
        print("="*40)

        # Prepare gallery data for saving
        gallery_data = {
            'embeddings': gallery_stack.cpu(),  # (G, 10, 768) tensor
            'labels': gallery_labels,           # List of class names
            'items': global_gallery_items,      # List of (class_name, path) tuples
            'classes': classes,                 # Sorted list of all classes
            'num_items': len(gallery_labels),   # Total number of gallery items
            'embedding_shape': gallery_stack.shape,
            'evaluation_config': {
                'eval_batch_size': eval_batch_size,
                'effective_eval_batch_size': effective_eval_batch_size,
                'num_gpus': num_gpus,
                'recall_ks': ks
            }
        }
        
        print(f" Gallery data prepared:")
        print(f"   - Embeddings: {gallery_stack.shape}")
        print(f"   - Items: {len(global_gallery_items)}")
        print(f"   - Classes: {len(classes)}")

        model.train()
        return hits[1] / total, mAP, gallery_data  # Return Recall@1, mAP, and gallery data

    # Initialize everything
    train_paths, test_paths, filename_to_class, path_to_class, dir_to_class = parse_instre(instre_root_path)
    
    # Debug: Print mapping statistics
    def print_mapping_stats(filename_to_class, path_to_class, dir_to_class, train_paths):
        """Print statistics about the filename and path mappings."""
        print("\n Mapping Statistics:")
        print(f"   - Filename mappings: {len(filename_to_class)}")
        print(f"   - Path mappings: {len(path_to_class)}")
        print(f"   - Directory mappings: {len(dir_to_class)}")
        
        # Count mappings per class
        class_counts = {}
        for path, class_name in path_to_class.items():
            class_counts[class_name] = class_counts.get(class_name, 0) + 1
        
        print(f"   - Classes with mappings: {len(class_counts)}")
        print(f"   - Average files per class: {sum(class_counts.values()) / len(class_counts):.1f}")
        
        # Verify consistency with train_paths
        inconsistencies = 0
        for class_name, paths in train_paths.items():
            for path in paths:
                if path in path_to_class:
                    if path_to_class[path] != class_name:
                        inconsistencies += 1
                        if inconsistencies <= 5:  # Show first 5 inconsistencies
                            print(f"     Inconsistency: {path} mapped to {path_to_class[path]} but in train_paths[{class_name}]")
                else:
                    inconsistencies += 1
                    if inconsistencies <= 5:
                        print(f"     Missing mapping: {path} in train_paths[{class_name}] but not in path_to_class")
        
        if inconsistencies == 0:
            print("    All path mappings are consistent with train_paths")
        else:
            print(f"     Found {inconsistencies} inconsistencies in path mappings")
        
        # Show some example mappings
        print("    Example path mappings:")
        for i, (path, class_name) in enumerate(list(path_to_class.items())[:5]):
            filename = Path(path).name
            print(f"      {filename} (from {path}) -> {class_name}")
    
    print_mapping_stats(filename_to_class, path_to_class, dir_to_class, train_paths)
    
    # Setup multi-GPU with DistributedDataParallel
    num_gpus = torch.cuda.device_count()
    print(f" Using {num_gpus} GPUs: {[torch.cuda.get_device_name(i) for i in range(num_gpus)]}")
    
    # Create clean model
    model = TrainableMultiVectorEncoder()
    
    if num_gpus > 1:
        # Initialize distributed training
        # For Modal's multi-GPU setup, DataParallel is actually more appropriate
        # than DDP since Modal gives us multiple GPUs in a single container/process
        device = torch.device("cuda:0")
        model = model.to(device)
        
        # Use DataParallel but with stability improvements
        model = nn.DataParallel(model)
        print(f" Model wrapped with DataParallel across {num_gpus} GPUs")
        effective_batch_size = batch_size * num_gpus
    else:
        device = torch.device("cuda:0")
        model = model.to(device)
        effective_batch_size = batch_size
    
    print(f"Primary device: {device}")
    print(f"Found {len(train_paths)} classes for training")
    print(f"Found {len(test_paths)} classes for testing")
    
    # Show some example classes
    example_classes = list(train_paths.keys())[:5]
    print(f"Example classes: {example_classes}")
    
    print(f"Model has {sum(p.numel() for p in model.parameters() if p.requires_grad):,} trainable parameters")
    print(f" Effective batch size: {effective_batch_size} (per_gpu: {batch_size}, gpus: {num_gpus})")
    
    # Setup optimizer and loss with stability improvements
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    #optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    
    # Use a stable scheduler instead of CosineAnnealingLR
    #scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.1)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.8, patience=20, min_lr=1e-7
    )
    
    criterion = TripletColbertLoss(margin=0.2)
    
    print(" Setting up efficient DataLoader...")
    
    # Make dataset large enough for multiple batches
    dataset_size = max(steps * batch_size * 2, 1000)  # Ensure enough data for all steps
    dataset = INSTRETripletDataset(train_paths, dir_to_class, steps_per_epoch=dataset_size)
    
    print(f" Dataset size: {dataset_size}, Batch size: {batch_size}, Expected batches: {dataset_size // batch_size}")
    
    # Use multiple workers for parallel image loading
    num_workers = min(4, batch_size)  # Don't exceed batch size
    dataloader = DataLoader(
        dataset, 
        batch_size=batch_size,
        shuffle=False,  # Dataset handles randomness internally
        num_workers=num_workers,
        pin_memory=True,  # Faster GPU transfer
        persistent_workers=True if num_workers > 0 else False,
    )
    
    print(f" DataLoader ready with {num_workers} workers")
    
    hist = []

    
    for i, (anchors, positives, negatives) in enumerate(tqdm(dataloader, desc="training", total=steps)):
        if i >= steps:
            break
            
        print(f" Step {i+1}/{steps}: Batch loaded by DataLoader")
        print(f"    Moving to device...")
        a, p, n = anchors.to(device), positives.to(device), negatives.to(device)

        print(f"    Forward pass...")
        optimizer.zero_grad()

        emb_a = model(a)
        emb_p = model(p)
        emb_n = model(n)
        loss = criterion(emb_a, emb_p, emb_n)
        
        print(f"    Backward pass...")
        loss.backward()
        print(f"Current Loss: {loss.item()}")
        print(f"Current LR: {scheduler.get_last_lr()[0]}")
        
        # Add gradient clipping for stability
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        
        optimizer.step()

        hist.append(loss.item())

        if (i + 1) % report_interval == 0:
            avg_loss = sum(hist[-report_interval:]) / report_interval
            print(f"[step {i+1:4d}] avg loss: {avg_loss:.4f} (effective_bs: {effective_batch_size})")

    # Final comprehensive evaluation
    print("--------------------------------")
    print(" Final evaluation:")
    print(f"Eval batch size: {eval_batch_size * num_gpus}")
    print(f"Final loss: {hist[-1]:.8f}")
    
    time.sleep(5) # For better printing!!
    
    # Run INSTRE-specific evaluation
    print(" Starting INSTRE retrieval evaluation...")
    recall_at_1, mAP, gallery_data = evaluate_retrieval_recalls(
        train_paths, test_paths, model, device,
        ks=[1, 2, 4, 8], eval_batch_size=eval_batch_size, num_gpus=num_gpus, 
        filename_to_class=filename_to_class, path_to_class=path_to_class, dir_to_class=dir_to_class
    )
    
    print(f" Multi-GPU ({num_gpus} GPUs) FP32 Training complete!")
    
    # ===== SAVE MODEL CHECKPOINT =====
    print(" Saving model checkpoint...")
    
    # Create checkpoint directory with timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    checkpoint_dir = Path(f"/mnt/data/Checkpoints/model_{timestamp}")
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    print(f" Created checkpoint directory: {checkpoint_dir}")
    
    # Prepare model for saving (unwrap DataParallel if needed)
    model_to_save = model.module if hasattr(model, 'module') else model
    
    # Save complete checkpoint
    checkpoint = {
        'model_state_dict': model_to_save.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict(),
        'training_history': hist,
        'final_loss': hist[-1] if hist else 0.0,
        'recall_at_1': recall_at_1,
        'mAP': mAP,
        'training_config': {
            'steps': steps,
            'batch_size': batch_size,
            'effective_batch_size': effective_batch_size,
            'lr': lr,
            'model_img_size': model_img_size,
            'num_gpus': num_gpus
        },
        'timestamp': timestamp
    }
    
    checkpoint_path = checkpoint_dir / "checkpoint.pth"
    torch.save(checkpoint, checkpoint_path)
    print(f" Saved complete checkpoint to: {checkpoint_path}")
    
    # Also save just the model weights for easier loading
    model_weights_path = checkpoint_dir / "model_weights.pth"
    torch.save(model_to_save.state_dict(), model_weights_path)
    print(f" Saved model weights to: {model_weights_path}")
    
    # Save gallery data
    print(" Saving gallery data...")
    gallery_path = checkpoint_dir / "gallery.pth"
    torch.save(gallery_data, gallery_path)
    print(f" Saved gallery data to: {gallery_path}")
    print(f"   - Gallery embeddings: {gallery_data['embedding_shape']}")
    print(f"   - Gallery items: {gallery_data['num_items']}")
    print(f"   - Gallery classes: {len(gallery_data['classes'])}")
    
    # Also save gallery metadata as text file for easy reference
    gallery_info_path = checkpoint_dir / "gallery_info.txt"
    with open(gallery_info_path, 'w') as f:
        f.write(f"Gallery Information - {timestamp}\n")
        f.write("=" * 40 + "\n")
        f.write(f"Total Gallery Items: {gallery_data['num_items']}\n")
        f.write(f"Embedding Shape: {gallery_data['embedding_shape']}\n")
        f.write(f"Number of Classes: {len(gallery_data['classes'])}\n")
        f.write(f"Evaluation Batch Size: {gallery_data['evaluation_config']['eval_batch_size']}\n")
        f.write(f"Effective Batch Size: {gallery_data['evaluation_config']['effective_eval_batch_size']}\n")
        f.write(f"Number of GPUs Used: {gallery_data['evaluation_config']['num_gpus']}\n")
        f.write(f"Recall Ks: {gallery_data['evaluation_config']['recall_ks']}\n")
        f.write("\nClasses:\n")
        for i, cls in enumerate(gallery_data['classes']):
            f.write(f"  {i+1:3d}. {cls}\n")
        f.write(f"\nFirst 10 Gallery Items:\n")
        for i, (class_name, path) in enumerate(gallery_data['items'][:10]):
            f.write(f"  {i+1:3d}. {Path(path).name} -> {class_name}\n")
    print(f" Saved gallery info to: {gallery_info_path}")
    
    # Save training config as text file for easy reference
    config_path = checkpoint_dir / "config.txt"
    with open(config_path, 'w') as f:
        f.write(f"Training Configuration - {timestamp}\n")
        f.write("=" * 40 + "\n")
        f.write(f"Steps: {steps}\n")
        f.write(f"Batch Size: {batch_size}\n")
        f.write(f"Effective Batch Size: {effective_batch_size}\n")
        f.write(f"Learning Rate: {lr}\n")
        f.write(f"Model Image Size: {model_img_size}\n")
        f.write(f"Number of GPUs: {num_gpus}\n")
        f.write(f"Final Loss: {hist[-1] if hist else 0.0:.6f}\n")
        f.write(f"Recall@1: {recall_at_1:.4f}\n")
        f.write(f"mAP: {mAP:.4f}\n")
        f.write(f"Total Training Steps: {len(hist)}\n")
    print(f" Saved training config to: {config_path}")
    
    print(f" All files saved to checkpoint directory: {checkpoint_dir}")
    print(f"    Files saved:")
    print(f"      - checkpoint.pth (complete checkpoint)")
    print(f"      - model_weights.pth (model weights only)")
    print(f"      - gallery.pth (gallery embeddings and metadata)")
    print(f"      - config.txt (training configuration)")
    print(f"      - gallery_info.txt (gallery information)")
    
    return {
        "final_loss": hist[-1] if hist else 0.0, 
        "avg_final_loss": sum(hist[-10:]) / 10 if len(hist) >= 10 else 0.0,
        "recall_at_1": recall_at_1,
        "mAP": mAP,
        "checkpoint_dir": str(checkpoint_dir),
        "timestamp": timestamp
    }


if __name__ == "__main__":
    # For local testing
    with app.run():
        result = main.remote()
        print("Result:", result) 
