"""
predict.py
Loads the trained U-Net and predicts a binary tumor mask for a single MRI image.

Usage
-----
    python src/predict.py --image path/to/scan.jpg --output mask_out.png
"""

import argparse
import os

import numpy as np
import torch
import albumentations as A
from albumentations.pytorch import ToTensorV2
from PIL import Image
import segmentation_models_pytorch as smp

from utils import IMG_SIZE

MODELS_DIR = "models"


def load_model(model_path: str, encoder: str, device):
    model = smp.Unet(encoder_name=encoder, encoder_weights=None, in_channels=3, classes=1)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()
    return model


def preprocess_image(image_path: str):
    transform = A.Compose([
        A.Resize(IMG_SIZE, IMG_SIZE),
        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ToTensorV2(),
    ])
    image = np.array(Image.open(image_path).convert("RGB"))
    augmented = transform(image=image)
    return augmented["image"].unsqueeze(0), image


def predict(image_path: str, output_path: str = None, model_path: str = None, encoder: str = "resnet34"):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_path = model_path or os.path.join(MODELS_DIR, "unet_brisc_seg.pth")
    model = load_model(model_path, encoder, device)

    input_tensor, original_image = preprocess_image(image_path)
    input_tensor = input_tensor.to(device)

    with torch.no_grad():
        prob_mask = torch.sigmoid(model(input_tensor))[0, 0].cpu().numpy()

    binary_mask = (prob_mask > 0.5).astype(np.uint8) * 255
    tumor_pixel_ratio = (prob_mask > 0.5).mean()

    print(f"Predicted tumor area: {tumor_pixel_ratio:.2%} of image")

    if output_path:
        Image.fromarray(binary_mask).resize(
            (original_image.shape[1], original_image.shape[0]), Image.NEAREST
        ).save(output_path)
        print(f"Mask saved to: {output_path}")

    return binary_mask, tumor_pixel_ratio


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Predict a tumor segmentation mask for an MRI image.")
    parser.add_argument("--image", required=True, help="Path to the MRI image")
    parser.add_argument("--output", default="predicted_mask.png", help="Path to save the predicted mask")
    parser.add_argument("--model", default=None, help="Path to a trained model checkpoint (optional)")
    parser.add_argument("--encoder", default="resnet34")
    args = parser.parse_args()

    predict(args.image, args.output, args.model, args.encoder)
