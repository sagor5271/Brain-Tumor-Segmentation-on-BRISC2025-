import os
import glob
import numpy as np
from PIL import Image
from torch.utils.data import Dataset
import albumentations as A
from albumentations.pytorch import ToTensorV2

IMG_SIZE = 256

class BriscSegDataset(Dataset):
    def __init__(self, root_dir, split="train", transform=None):
        self.images_dir = os.path.join(
            root_dir, "segmentation_task", split, "images"
        )
        self.masks_dir = os.path.join(
            root_dir, "segmentation_task", split, "masks"
        )
        self.image_paths = sorted(
            glob.glob(os.path.join(self.images_dir, "*.jpg"))
        )
        self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    def _mask_path_for(self, image_path):
        basename = os.path.splitext(os.path.basename(image_path))[0]
        return os.path.join(self.masks_dir, basename + ".png")

    def __getitem__(self, idx):
        image_path = self.image_paths[idx]
        mask_path = self._mask_path_for(image_path)

        image = np.array(Image.open(image_path).convert("RGB"))
        mask = np.array(Image.open(mask_path).convert("L"))

        if self.transform:
            augmented = self.transform(image=image, mask=mask)
            image = augmented["image"]
            mask = augmented["mask"]

        mask = (mask > 0).float().unsqueeze(0)
        return image, mask


def get_train_transform(img_size=IMG_SIZE):
    return A.Compose([
        A.Resize(img_size, img_size),
        A.HorizontalFlip(p=0.5),
        A.Rotate(limit=15, p=0.5),
        A.RandomBrightnessContrast(p=0.3),
        A.Normalize(
            mean=(0.485, 0.456, 0.406),
            std=(0.229, 0.224, 0.225)
        ),
        ToTensorV2(),
    ])


def get_test_transform(img_size=IMG_SIZE):
    return A.Compose([
        A.Resize(img_size, img_size),
        A.Normalize(
            mean=(0.485, 0.456, 0.406),
            std=(0.229, 0.224, 0.225)
        ),
        ToTensorV2(),
    ])
