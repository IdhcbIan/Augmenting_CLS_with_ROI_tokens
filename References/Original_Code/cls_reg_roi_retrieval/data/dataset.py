import os, glob, random
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms
from cls_reg_roi_retrieval.config import CFG

_t = transforms.Compose([
    transforms.Resize(256),
    transforms.RandomResizedCrop(CFG.img_size),
    transforms.ToTensor(),
])

class TripletFolder(Dataset):
    def __init__(self, root):
        self.root = root
        self.classes = sorted(os.listdir(root))
        self.imgs = {c: glob.glob(os.path.join(root, c, "*"))
                     for c in self.classes}

    def _sample(self, cls): return _t(Image.open(random.choice(self.imgs[cls])).convert("RGB"))

    def __len__(self): return 10_000_000  # iterable

    def __getitem__(self, _):
        pos_cls = random.choice(self.classes)
        neg_cls = random.choice([c for c in self.classes if c != pos_cls])
        return self._sample(pos_cls), self._sample(pos_cls), self._sample(neg_cls)
