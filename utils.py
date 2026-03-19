"""
U-Net Training Utilities
========================
Shared utilities for DRIVE dataset retinal vessel segmentation project.
Includes dataset loader, model loading, visualization helpers, and debugging tools.
Used by train.py and test.py scripts.

Key components:
- DriveDataset: Loads image/mask pairs with automatic resizing and normalization
- Model checkpoint loading for U-Net
- Visualization utilities (denormalize for matplotlib)
- Dataset statistics checker
"""

import os
import glob
import torch
from torch.utils.data import Dataset
import cv2
import numpy as np
import matplotlib.pyplot as plt
from UNet import UNet

class DriveDataset(Dataset):
    """
    DRIVE dataset loader for retinal vessel segmentation training/inference.

    Expects processed data structure:
    processed_data/
        train/images/, train/masks/
        val/images/, val/masks/

    Handles both training (image+mask) and inference (image only) modes.
    """
    def __init__(self, images_path, masks_path=None, size=(512, 512)):
        self.images = images_path
        self.masks = masks_path
        self.size = size

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        # Load and preprocess image
        img_path = self.images[idx]
        image = cv2.imread(str(img_path))
        if image is None:
            raise ValueError(f"Cannot load image: {img_path}")
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = cv2.resize(image, self.size)
        image = image.astype(np.float32) / 255.0
        image = torch.from_numpy(image).permute(2, 0, 1)  # HWC -> CHW

        # Load mask if available
        if self.masks is not None:
            mask_path = self.masks[idx]
            mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
            if mask is None:
                raise ValueError(f"Cannot load mask: {mask_path}")
            mask = cv2.resize(mask, self.size)
            mask = mask.astype(np.float32) / 255.0
            mask = torch.from_numpy(mask).unsqueeze(0)  # Add channel dim
            return image, mask
        return image

def seeding(seed=42):
    """Set random seeds across torch, numpy for reproducible results."""
    torch.manual_seed(seed)
    np.random.seed(seed)

def create_dir(path):
    """Create directory if it doesn't exist."""
    if not os.path.exists(path):
        os.makedirs(path)

def epoch_time(start_time, end_time):
    """Convert elapsed time to (minutes, seconds) format."""
    elapsed_time = end_time - start_time
    mins = int(elapsed_time / 60)
    secs = int(elapsed_time % 60)
    return mins, secs

def denormalize(tensor):
    """
    Convert normalized tensor back to displayable image format.
    Works with both RGB (3,H,W) and grayscale (1,H,W) tensors.
    """
    img = tensor.squeeze().cpu().numpy()

    # Handle different tensor shapes
    if img.ndim == 3 and img.shape[0] in [1, 3, 4]:
        if img.shape[0] == 3:  # RGB: (3,H,W) -> (H,W,3)
            img = np.transpose(img, (1, 2, 0))
        elif img.shape[0] == 1:  # Grayscale: (1,H,W) -> (H,W)
            img = img[0]
    elif img.ndim == 2:  # Already (H,W)
        pass
    else:
        raise ValueError(f"Unsupported tensor shape: {img.shape}")

    # Convert to uint8 range [0,255]
    img = np.clip(img * 255, 0, 255).astype(np.uint8)
    return img

def load_model_checkpoint(checkpoint_path, device):
    """
    Load pretrained U-Net weights from checkpoint file.

    Args:
        checkpoint_path: Path to .pth model weights
        device: torch.device('cuda' or 'cpu')

    Returns:
        Loaded U-Net model ready for inference/training
    """
    model = UNet(in_channels=3, out_channels=1, features=[64, 128, 256, 512]).to(device)
    if os.path.exists(checkpoint_path):
        checkpoint = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(checkpoint)
        print(f"Model loaded from: {checkpoint_path}")
        return model
    else:
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

def check_dataset(data_dir="processed_data"):
    """
    Quick dataset verification function.
    Prints file counts and mask statistics for debugging.
    """
    train_x = sorted(glob.glob(f"{data_dir}/train/images/*.png"))
    train_y = sorted(glob.glob(f"{data_dir}/train/masks/*.png"))
    valid_x = sorted(glob.glob(f"{data_dir}/val/images/*.png"))
    valid_y = sorted(glob.glob(f"{data_dir}/val/masks/*.png"))

    print("Dataset overview:")
    print(f"  Train images: {len(train_x)}")
    print(f"  Train masks:  {len(train_y)}")
    print(f"  Valid images: {len(valid_x)}")
    print(f"  Valid masks:  {len(valid_y)}")

    # Check mask properties
    if len(train_y) > 0:
        sample_mask = cv2.imread(train_y[0], cv2.IMREAD_GRAYSCALE)
        if sample_mask is not None:
            vascular_ratio = np.mean(sample_mask > 127)
            print(f"  Sample mask vascular pixels: {vascular_ratio:.2%}")
            print(f"  Mask pixel range (raw): {sample_mask.min()} - {sample_mask.max()}")

            # Normalized range (as seen by DriveDataset)
            normalized_mask = sample_mask.astype(np.float32) / 255.0
            print(f"  Mask range (normalized): {normalized_mask.min():.3f} - {normalized_mask.max():.3f}")