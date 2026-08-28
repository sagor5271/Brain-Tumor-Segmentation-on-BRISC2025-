"""
train.py
Trains a U-Net (ResNet34 encoder, ImageNet-pretrained) for binary brain
tumor segmentation on the BRISC2025 dataset.

Usage
-----
    python src/train.py --data_root /kaggle/input/datasets/briscdataset/brisc2025/brisc2025 \
                         --epochs 20 --batch_size 16
"""

import os
import argparse
import json

import torch
import albumentations as A
from albumentations.pytorch import ToTensorV2
from torch.utils.data import DataLoader
import segmentation_models_pytorch as smp

from utils import BriscSegDataset, DiceBCELoss, dice_coefficient, iou_score, \
    plot_training_curves, plot_sample_predictions, IMG_SIZE

MODELS_DIR = "models"
RESULTS_DIR = "results"


def get_transforms():
    train_tf = A.Compose([
        A.Resize(IMG_SIZE, IMG_SIZE),
        A.HorizontalFlip(p=0.5),
        A.Rotate(limit=15, p=0.5),
        A.RandomBrightnessContrast(p=0.3),
        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ToTensorV2(),
    ])
    test_tf = A.Compose([
        A.Resize(IMG_SIZE, IMG_SIZE),
        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ToTensorV2(),
    ])
    return train_tf, test_tf


def run_epoch(model, loader, criterion, device, optimizer=None):
    is_train = optimizer is not None
    model.train() if is_train else model.eval()

    total_loss, total_dice, total_iou, n_batches = 0.0, 0.0, 0.0, 0

    context = torch.enable_grad() if is_train else torch.no_grad()
    with context:
        for images, masks in loader:
            images, masks = images.to(device), masks.to(device)

            if is_train:
                optimizer.zero_grad()

            logits = model(images)
            loss = criterion(logits, masks)

            if is_train:
                loss.backward()
                optimizer.step()

            probs = torch.sigmoid(logits)
            total_loss += loss.item()
            total_dice += dice_coefficient(probs, masks)
            total_iou += iou_score(probs, masks)
            n_batches += 1

    return total_loss / n_batches, total_dice / n_batches, total_iou / n_batches


def main():
    parser = argparse.ArgumentParser(description="Train a U-Net brain tumor segmentation model on BRISC2025.")
    parser.add_argument("--data_root", required=True,
                         help="Path to the brisc2025 root folder (containing segmentation_task/)")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--encoder", default="resnet34")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    train_tf, test_tf = get_transforms()
    train_dataset = BriscSegDataset(args.data_root, split="train", transform=train_tf)
    test_dataset = BriscSegDataset(args.data_root, split="test", transform=test_tf)

    print(f"Train samples: {len(train_dataset)} | Test samples: {len(test_dataset)}")

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=2)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=2)

    model = smp.Unet(
        encoder_name=args.encoder,
        encoder_weights="imagenet",
        in_channels=3,
        classes=1,
    ).to(device)

    criterion = DiceBCELoss(bce_weight=0.5)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=3)

    history = {"train_loss": [], "val_loss": [], "train_dice": [], "val_dice": []}
    best_val_dice = 0.0

    os.makedirs(MODELS_DIR, exist_ok=True)
    os.makedirs(RESULTS_DIR, exist_ok=True)

    for epoch in range(1, args.epochs + 1):
        train_loss, train_dice, train_iou = run_epoch(model, train_loader, criterion, device, optimizer)
        val_loss, val_dice, val_iou = run_epoch(model, test_loader, criterion, device)

        scheduler.step(val_loss)

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["train_dice"].append(train_dice)
        history["val_dice"].append(val_dice)

        print(f"Epoch {epoch}/{args.epochs} | "
              f"Train Loss: {train_loss:.4f} Dice: {train_dice:.4f} IoU: {train_iou:.4f} | "
              f"Val Loss: {val_loss:.4f} Dice: {val_dice:.4f} IoU: {val_iou:.4f}")

        if val_dice > best_val_dice:
            best_val_dice = val_dice
            torch.save(model.state_dict(), os.path.join(MODELS_DIR, "unet_brisc_seg.pth"))

    with open(os.path.join(RESULTS_DIR, "training_history.json"), "w") as f:
        json.dump(history, f, indent=2)

    plot_training_curves(history, save_path=os.path.join(RESULTS_DIR, "loss_dice_curves.png"))

    # Qualitative check: visualize predictions on a batch from the test set
    model.load_state_dict(torch.load(os.path.join(MODELS_DIR, "unet_brisc_seg.pth"), map_location=device))
    model.eval()
    images, masks = next(iter(test_loader))
    with torch.no_grad():
        preds = torch.sigmoid(model(images.to(device))).cpu()
    plot_sample_predictions(images, masks, preds, save_path=os.path.join(RESULTS_DIR, "sample_predictions.png"))

    print(f"\nBest validation Dice: {best_val_dice:.4f}")
    print(f"Model saved to: {MODELS_DIR}/unet_brisc_seg.pth")


if __name__ == "__main__":
    main()
