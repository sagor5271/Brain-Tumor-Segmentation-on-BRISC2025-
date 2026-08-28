import torch
import segmentation_models_pytorch as smp

def build_model():
    return smp.Unet(
        encoder_name="resnet34",
        encoder_weights="imagenet",
        in_channels=3,
        classes=1,
    )

def load_model(device):
    model = build_model().to(device)
    return model
