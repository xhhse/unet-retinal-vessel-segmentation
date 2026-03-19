# test.py
"""
U-Net Model Evaluation Script
=============================
Evaluates trained U-Net on validation set with Dice coefficient.
Generates per-image visualizations and summary statistics.

Expected inputs:
- Model weights: files/unet_checkpoint.pth
- Test data: processed_data/val/images/*.png + masks/*.png
"""

import torch
import glob
import matplotlib.pyplot as plt
import numpy as np
import cv2
import os
from torch.utils.data import DataLoader
import pandas as pd
from utils import DriveDataset, create_dir, denormalize, load_model_checkpoint


def dice_coefficient(pred, target, threshold=0.5, smooth=1e-6):
    """
    Calculate Dice coefficient between predicted and ground truth masks.

    Args:
        pred: Model output after sigmoid (0-1 range)
        target: Ground truth mask (0-1 range)
        threshold: Binary threshold for prediction
        smooth: Smoothing factor to avoid division by zero

    Returns:
        Dice score (0-1, higher is better)
    """
    pred_binary = (pred > threshold).float()
    target_binary = (target > 0.5).float()

    intersection = (pred_binary * target_binary).sum()
    union = pred_binary.sum() + target_binary.sum()

    dice = (2 * intersection + smooth) / (union + smooth)
    return dice.item()


if __name__ == "__main__":
    print("U-Net Model Evaluation - Retinal Vessel Segmentation")
    print("=" * 60)

    # Setup output directory
    create_dir("test_results")

    # Load test data (using validation set for quick evaluation)
    test_x = sorted(glob.glob("processed_data/val/images/*.png"))
    test_y = sorted(glob.glob("processed_data/val/masks/*.png"))

    print(f"Evaluating {len(test_x)} test images")

    # Device setup
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Load trained model
    model = load_model_checkpoint("files/unet_checkpoint.pth", device)
    model.eval()

    # Create test dataloader
    test_dataset = DriveDataset(test_x, test_y)
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)

    # Track metrics
    dice_scores = []

    print("\nProcessing images...")
    print("-" * 40)

    # Inference loop with visualization
    with torch.no_grad():
        for idx, (images, masks) in enumerate(test_loader):
            images, masks = images.to(device), masks.to(device)
            preds = torch.sigmoid(model(images))

            # Calculate Dice score
            dice_score = dice_coefficient(preds[0], masks[0])
            dice_scores.append(dice_score)

            # Convert tensors to numpy for visualization
            image_np = denormalize(images[0])
            mask_np = denormalize(masks[0])
            pred_np = denormalize(preds[0])

            # Create RGB overlay for comparison
            overlay = np.zeros((512, 512, 3), dtype=np.uint8)
            overlay[..., 0] = pred_np * 255   # Red channel = prediction
            overlay[..., 1] = mask_np * 255   # Green channel = ground truth

            # 6-panel visualization
            fig, axes = plt.subplots(2, 3, figsize=(15, 10))

            axes[0,0].imshow(image_np)
            axes[0,0].set_title('Original Image')
            axes[0,0].axis('off')

            axes[0,1].imshow(mask_np, cmap='gray')
            axes[0,1].set_title('Ground Truth')
            axes[0,1].axis('off')

            axes[0,2].imshow(pred_np, cmap='gray')
            axes[0,2].set_title(f'Prediction\nDice: {dice_score:.3f}')
            axes[0,2].axis('off')

            axes[1,0].imshow(image_np)
            axes[1,0].imshow(pred_np, cmap='jet', alpha=0.5)
            axes[1,0].set_title('Image + Prediction')
            axes[1,0].axis('off')

            axes[1,1].imshow(image_np)
            axes[1,1].imshow(mask_np, cmap='Greens', alpha=0.5)
            axes[1,1].set_title('Image + Ground Truth')
            axes[1,1].axis('off')

            axes[1,2].imshow(overlay)
            axes[1,2].set_title('Overlay\n(Red=Pred, Green=GT)')
            axes[1,2].axis('off')

            plt.tight_layout()
            plt.savefig(f"test_results/test_{idx+1:03d}.png", dpi=150, bbox_inches='tight')
            plt.close()

            print(f"Image {idx+1:2d}: Dice = {dice_score:.4f}")

            if (idx + 1) % 5 == 0:
                print(f"Progress: {idx+1}/{len(test_loader)} images")

    # Summary statistics
    print("\n" + "="*60)
    print("EVALUATION SUMMARY")
    print("="*60)
    print(f"Total images tested:      {len(dice_scores)}")
    print(f"Mean Dice coefficient:    {np.mean(dice_scores):.4f}")
    print(f"Dice standard deviation:  {np.std(dice_scores):.4f}")
    print(f"Best Dice score:          {np.max(dice_scores):.4f}")
    print(f"Worst Dice score:         {np.min(dice_scores):.4f}")

    # Performance interpretation
    mean_dice = np.mean(dice_scores)
    if mean_dice > 0.80:
        print("✓ Excellent performance (publication quality)")
    elif mean_dice > 0.75:
        print("✓ Good performance (state-of-the-art range)")
    elif mean_dice > 0.70:
        print("○ Acceptable (room for improvement)")
    else:
        print("⚠ Needs more training or hyperparameter tuning")

    # Save detailed results
    metrics_df = pd.DataFrame({
        'image_id': [f'test_{i+1:03d}' for i in range(len(dice_scores))],
        'dice_score': dice_scores
    })
    metrics_df.to_csv('test_results/dice_scores.csv', index=False)

    print(f"\nDetailed results saved to: test_results/dice_scores.csv")
    print(f"Individual visualizations: test_results/test_*.png")
    print("\nEvaluation complete!")
