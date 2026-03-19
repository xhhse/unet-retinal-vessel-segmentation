"""
U-Net Training Script for Retinal Vessel Segmentation
=====================================================
Training pipeline for DRIVE dataset using pure Dice Loss.
Optimized hyperparameters for vessel segmentation task.

Expected dataset structure (from preprocess_drive.py):
processed_data/
├── train/    (10 original images × augmentations = 40-60 images)
├── val/      (10 images, no augmentation)  
└── test/     (30 images, no augmentation)
"""

import os
import time
import glob
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import pandas as pd

from UNet import UNet
from utils import DriveDataset, seeding, create_dir, epoch_time, check_dataset


# ============================================================
# Loss Functions
# ============================================================
class DiceLoss(nn.Module):
    """Dice Loss optimized for imbalanced vessel segmentation."""
    def __init__(self, smooth=1.0):
        super().__init__()
        self.smooth = smooth

    def forward(self, pred, target):
        pred = torch.sigmoid(pred)
        inter = (pred * target).sum(dim=(2, 3))
        dice = (2 * inter + self.smooth) / (pred.sum(dim=(2, 3)) + target.sum(dim=(2, 3)) + self.smooth)
        return 1 - dice.mean()


class FocalDiceLoss(nn.Module):
    """Combined Focal + Dice loss (alternative, not used in main training)."""
    def __init__(self, alpha=0.1, gamma=2.0, smooth=1.0, dice_weight=3.0):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.smooth = smooth
        self.dice_weight = dice_weight

    def focal_loss(self, pred, target):
        pred_sigmoid = torch.sigmoid(pred)
        pt = torch.where(target == 1, pred_sigmoid, 1 - pred_sigmoid)
        focal_weight = self.alpha * (1 - pt) ** self.gamma
        return focal_weight * F.binary_cross_entropy_with_logits(pred, target, reduction='none')

    def dice_loss(self, pred, target):
        pred = torch.sigmoid(pred)
        inter = (pred * target).sum(dim=(2, 3))
        dice = (2 * inter + self.smooth) / (pred.sum(dim=(2, 3)) + target.sum(dim=(2, 3)) + self.smooth)
        return 1 - dice.mean()

    def forward(self, pred, target):
        focal = self.focal_loss(pred, target).mean()
        dice = self.dice_loss(pred, target)
        return focal + self.dice_weight * dice


# ============================================================
# Training Functions
# ============================================================
def train_epoch(model, loader, optimizer, loss_fn, device):
    """Single training epoch."""
    epoch_loss = 0.0
    model.train()
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        optimizer.zero_grad()
        y_pred = model(x)
        loss = loss_fn(y_pred, y)
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item()
    return epoch_loss / len(loader)


def validate_epoch(model, loader, loss_fn, device):
    """Single validation epoch."""
    epoch_loss = 0.0
    model.eval()
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            y_pred = model(x)
            loss = loss_fn(y_pred, y)
            epoch_loss += loss.item()
    return epoch_loss / len(loader)


# ============================================================
# Main Training Pipeline
# ============================================================
if __name__ == "__main__":
    print("U-Net Training - Retinal Vessel Segmentation")
    print("=" * 60)

    # Setup
    seeding(42)
    create_dir("files")
    check_dataset()

    # Load dataset paths
    train_x = sorted(glob.glob("processed_data/train/images/*.png"))
    train_y = sorted(glob.glob("processed_data/train/masks/*.png"))
    valid_x = sorted(glob.glob("processed_data/val/images/*.png"))
    valid_y = sorted(glob.glob("processed_data/val/masks/*.png"))

    print(f"Dataset loaded — Train: {len(train_x)}, Valid: {len(valid_x)}")

    # Hyperparameters (tuned for DRIVE vessel segmentation)
    H, W = 512, 512
    batch_size = 4
    num_epochs = 200
    lr = 2e-4

    # Data loaders
    train_dataset = DriveDataset(train_x, train_y, (H, W))
    valid_dataset = DriveDataset(valid_x, valid_y, (H, W))

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True,
                              num_workers=2, pin_memory=False)
    valid_loader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=False,
                              num_workers=2, pin_memory=False)

    # Model setup
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    model = UNet(in_channels=3, out_channels=1, features=[64, 128, 256, 512]).to(device)

    # Load existing checkpoint if available
    if os.path.exists("files/unet_checkpoint.pth"):
        checkpoint = torch.load("files/unet_checkpoint.pth", map_location=device)
        model.load_state_dict(checkpoint)
        print("Loaded existing best model checkpoint")

    # Optimizer and loss
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=10, factor=0.5)
    loss_fn = DiceLoss(smooth=1.0)

    # Training tracking
    train_losses, val_losses = [], []
    best_valid_loss = float("inf")

    print("\nTraining Progress:")
    print("-" * 60)
    print(f"{'Epoch':<6} {'Train':<8} {'Val':<8} {'Time':<8} {'Status'}")
    print("-" * 60)

    # Training loop
    for epoch in range(num_epochs):
        start_time = time.time()

        train_loss = train_epoch(model, train_loader, optimizer, loss_fn, device)
        valid_loss = validate_epoch(model, valid_loader, loss_fn, device)

        train_losses.append(train_loss)
        val_losses.append(valid_loss)

        # Learning rate scheduling (silent)
        scheduler.step(valid_loss)

        # Save best model
        if valid_loss < best_valid_loss:
            best_valid_loss = valid_loss
            torch.save(model.state_dict(), "files/unet_checkpoint.pth")
            status = "BEST"
        else:
            status = ""

        # Progress logging - WITHOUT LR
        epoch_mins, epoch_secs = epoch_time(start_time, time.time())
        print(f"{epoch + 1:<6} {train_loss:<8.4f} {valid_loss:<8.4f} "
              f"{epoch_mins}m{epoch_secs:02d}s {status}")

        # Periodic full checkpoints
        if (epoch + 1) % 20 == 0:
            checkpoint = {
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': valid_loss,
            }
            torch.save(checkpoint, f"files/checkpoint_epoch_{epoch + 1:03d}.pth")

    # ============================================================
    # Results and Visualization
    # ============================================================
    print("\n" + "=" * 60)
    print("Training completed successfully!")
    print("=" * 60)
    print(f"Best validation loss: {best_valid_loss:.4f}")
    print(f"Expected Dice coefficient: ~{1 - best_valid_loss:.3f}")

    # Save training history
    history_df = pd.DataFrame({
        'epoch': range(1, num_epochs + 1),
        'train_loss': train_losses,
        'val_loss': val_losses
    })
    history_df.to_csv('files/training_history.csv', index=False)

    # Plot training curves
    plt.figure(figsize=(15, 5))

    plt.subplot(1, 3, 1)
    plt.plot(train_losses, label='Train Loss', linewidth=2)
    plt.plot(val_losses, label='Validation Loss', linewidth=2)
    plt.xlabel('Epoch')
    plt.ylabel('Dice Loss')
    plt.title('Training Curves')
    plt.legend()
    plt.grid(True, alpha=0.3)

    plt.subplot(1, 3, 2)
    plt.semilogy(train_losses, label='Train', linewidth=2)
    plt.semilogy(val_losses, label='Validation', linewidth=2)
    plt.xlabel('Epoch')
    plt.ylabel('Loss (log scale)')
    plt.title('Loss (Log Scale)')
    plt.legend()
    plt.grid(True, alpha=0.3)

    plt.subplot(1, 3, 3)
    plt.axhline(y=best_valid_loss, color='r', linestyle='--', 
                label=f'Best: {best_valid_loss:.3f}')
    plt.plot(val_losses, 'g-', linewidth=2, label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Model Convergence')
    plt.legend()
    plt.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('files/loss_curves.png', dpi=150, bbox_inches='tight')
    plt.show()

    print("\nOutput files:")
    print("  • unet_checkpoint.pth (best model weights)")
    print("  • training_history.csv (loss tracking)")
    print("  • loss_curves.png (visualization)")
    print("  • checkpoint_epoch_XXX.pth (periodic backups)")