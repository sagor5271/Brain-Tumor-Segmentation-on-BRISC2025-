"""
utils.py
Shared helpers for the BRISC2025 Brain Tumor Segmentation project.

Expected dataset layout (BRISC2025, segmentation_task):
    <DATASET_ROOT>/segmentation_task/train/images/*.jpg
    <DATASET_ROOT>/segmentation_task/train/masks/*.png
    <DATASET_ROOT>/segmentation_task/test/images/*.jpg
    <DATASET_ROOT>/segmentation_task/test/masks/*.png

Image and mask filenames share the same basename
(e.g. brisc2025_train_00001_gl_ax_t1.jpg <-> brisc2025_train_00001_gl_ax_t1.png).
Masks are binary (tumor vs. background).
"""

import os
import glob

import numpy as np
import torch
import matplotlib.pyplot as plt
from PIL import Image
from torch.utils.data import Dataset

IMG_SIZE = 256


class BriscSegDataset(Dataset):
    """PyTorch Dataset for the BRISC2025 segmentation_task split."""

    def __init__(self, root_dir: str, split: str = "train", transform=None):
        self.images_dir = os.path.join(root_dir, "segmentation_task", split, "images")
        self.masks_dir = os.path.join(root_dir, "segmentation_task", split, "masks")

        self.image_paths = sorted(glob.glob(os.path.join(self.images_dir, "*.jpg")))
        if len(self.image_paths) == 0:
            # fall back in case of .png images
            self.image_paths = sorted(glob.glob(os.path.join(self.images_dir, "*.png")))

        self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    def _mask_path_for(self, image_path: str) -> str:
        basename = os.path.splitext(os.path.basename(image_path))[0]
        return os.path.join(self.masks_dir, basename + ".png")

    def __getitem__(self, idx):
        image_path = self.image_paths[idx]
        mask_path = self._mask_path_for(image_path)

        image = Image.open(image_path).convert("RGB")
        mask = Image.open(mask_path).convert("L")  # single-channel binary mask

        if self.transform:
            augmented = self.transform(image=np.array(image), mask=np.array(mask))
            image = augmented["image"]
            mask = augmented["mask"]

        # Binarize mask (0/1) and add channel dimension
        mask = (mask > 0).float().unsqueeze(0) if torch.is_tensor(mask) else torch.from_numpy((mask > 0).astype("float32")).unsqueeze(0)

        return image, mask


def dice_coefficient(pred, target, eps: float = 1e-6):
    """Dice score for binary segmentation. pred/target: (N, 1, H, W) tensors of 0/1 or probabilities."""
    pred = (pred > 0.5).float()
    intersection = (pred * target).sum(dim=(1, 2, 3))
    union = pred.sum(dim=(1, 2, 3)) + target.sum(dim=(1, 2, 3))
    dice = (2 * intersection + eps) / (union + eps)
    return dice.mean().item()


def iou_score(pred, target, eps: float = 1e-6):
    """IoU (Jaccard index) for binary segmentation."""
    pred = (pred > 0.5).float()
    intersection = (pred * target).sum(dim=(1, 2, 3))
    union = pred.sum(dim=(1, 2, 3)) + target.sum(dim=(1, 2, 3)) - intersection
    iou = (intersection + eps) / (union + eps)
    return iou.mean().item()


class DiceBCELoss(torch.nn.Module):
    """Combined Dice + BCE loss — standard choice for imbalanced medical segmentation masks."""

    def __init__(self, bce_weight: float = 0.5):
        super().__init__()
        self.bce = torch.nn.BCEWithLogitsLoss()
        self.bce_weight = bce_weight

    def forward(self, logits, target):
        bce_loss = self.bce(logits, target)

        probs = torch.sigmoid(logits)
        intersection = (probs * target).sum(dim=(1, 2, 3))
        union = probs.sum(dim=(1, 2, 3)) + target.sum(dim=(1, 2, 3))
        dice_loss = 1 - ((2 * intersection + 1e-6) / (union + 1e-6)).mean()

        return self.bce_weight * bce_loss + (1 - self.bce_weight) * dice_loss


def plot_training_curves(history, save_path=None):
    epochs = range(1, len(history["train_loss"]) + 1)
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))

    axes[0].plot(epochs, history["train_loss"], marker="o", label="Train Loss")
    axes[0].plot(epochs, history["val_loss"], marker="o", label="Val Loss")
    axes[0].set_title("Loss per Epoch")
    axes[0].set_xlabel("Epoch")
    axes[0].legend()
    axes[0].grid(alpha=0.3)

    axes[1].plot(epochs, history["train_dice"], marker="o", label="Train Dice")
    axes[1].plot(epochs, history["val_dice"], marker="o", label="Val Dice")
    axes[1].set_title("Dice Score per Epoch")
    axes[1].set_xlabel("Epoch")
    axes[1].legend()
    axes[1].grid(alpha=0.3)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150)
    plt.close(fig)


def plot_sample_predictions(images, masks, preds, save_path=None, n=4):
    """images, masks, preds: batches of tensors (N, C, H, W)."""
    n = min(n, images.size(0))
    fig, axes = plt.subplots(n, 3, figsize=(9, 3 * n))
    if n == 1:
        axes = axes.reshape(1, -1)

    for i in range(n):
        img = images[i].permute(1, 2, 0).cpu().numpy()
        img = (img - img.min()) / (img.max() - img.min() + 1e-8)
        gt = masks[i, 0].cpu().numpy()
        pred = (preds[i, 0].cpu().numpy() > 0.5).astype(float)

        axes[i, 0].imshow(img)
        axes[i, 0].set_title("MRI Image")
        axes[i, 1].imshow(gt, cmap="gray")
        axes[i, 1].set_title("Ground Truth Mask")
        axes[i, 2].imshow(pred, cmap="gray")
        axes[i, 2].set_title("Predicted Mask")
        for ax in axes[i]:
            ax.axis("off")

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150)
    plt.close(fig)
