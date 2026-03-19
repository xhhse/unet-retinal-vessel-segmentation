"""
DRIVE Dataset Preprocessor
--------------------------------------------------
This script handles preprocessing for the DRIVE retinal vessel segmentation dataset.
It was written to streamline common preparation tasks before model training, including:

1. Reading images and manual masks from the original DRIVE dataset folders.
2. Splitting training data into training and validation subsets.
3. Applying geometric and photometric augmentations such as flipping, rotation, and brightness/contrast changes.
4. Resizing all outputs to a consistent size (default 512x512).
5. Saving the processed images and masks in PNG format for easier downstream use.

Example:
    python drive_dataset_preprocessor.py --input ./DRIVE --output ./processed_data --augmentations 6 --resize 256 256

Notes:
- The first 10 training samples are used as validation by default.
- Any output size can be specified via arguments.
- Augmentation count controls how many transformed samples are generated per image.
"""

import argparse
import numpy as np
import cv2
from glob import glob
from tqdm import tqdm
from pathlib import Path
import albumentations as A
from typing import Tuple, List
from dataclasses import dataclass


@dataclass
class AugmentationConfig:
    """Stores parameters for data augmentation"""
    size: Tuple[int, int] = (512, 512)
    flip_prob: float = 0.5
    rotate_limit: int = 15
    brightness_limit: Tuple[float, float] = (0.9, 1.1)
    contrast_limit: Tuple[float, float] = (0.9, 1.1)


class DRIVEPreprocessor:
    """Preprocessor class for the DRIVE dataset with configurable augmentation and resize options"""

    def __init__(self, data_path: str, output_path: str,
                 n_augmentations: int = 4, resize_dim: Tuple[int, int] = (512, 512)):
        self.data_path = Path(data_path)
        self.output_path = Path(output_path)
        self.config = AugmentationConfig(size=resize_dim)
        self.n_augmentations = n_augmentations

    def create_directories(self) -> None:
        """Create the necessary output directory structure"""
        dirs = [
            self.output_path / "train" / "images",
            self.output_path / "train" / "masks",
            self.output_path / "val" / "images",
            self.output_path / "val" / "masks",
            self.output_path / "test" / "images",
            self.output_path / "test" / "masks"
        ]
        for d in dirs:
            d.mkdir(parents=True, exist_ok=True)

    def load_dataset_paths(self) -> dict:
        """Load image and mask paths, then split into train/val/test"""
        paths = {}
        paths['train_images'] = sorted(glob(str(self.data_path / "training" / "images" / "*.tif")))
        paths['train_masks'] = sorted(glob(str(self.data_path / "training" / "1st_manual" / "*.gif")))
        paths['test_images'] = sorted(glob(str(self.data_path / "test" / "images" / "*.tif")))
        paths['test_masks'] = sorted(glob(str(self.data_path / "test" / "1st_manual" / "*.gif")))

        split_idx = 10
        paths['val_images'] = paths['train_images'][:split_idx]
        paths['val_masks'] = paths['train_masks'][:split_idx]
        paths['train_images'] = paths['train_images'][split_idx:]
        paths['train_masks'] = paths['train_masks'][split_idx:]

        print(f"Dataset split — Train: {len(paths['train_images'])}, Val: {len(paths['val_images'])}, Test: {len(paths['test_images'])}")
        return paths

    def get_augmentation_pipeline(self, augment=True) -> A.Compose:
        """Build the Albumentations augmentation pipeline"""
        resize_transform = A.Resize(*self.config.size, p=1.0)
        if not augment:
            return A.Compose([resize_transform], is_check_shapes=False)

        return A.Compose([
            A.HorizontalFlip(p=self.config.flip_prob),
            A.VerticalFlip(p=self.config.flip_prob),
            A.Rotate(limit=self.config.rotate_limit, p=self.config.flip_prob),
            A.RandomBrightnessContrast(
                brightness_limit=self.config.brightness_limit,
                contrast_limit=self.config.contrast_limit,
                p=self.config.flip_prob
            ),
            resize_transform
        ], is_check_shapes=False)

    def load_image_mask_pair(self, img_path, mask_path):
        """Load an image and its corresponding mask"""
        image = cv2.imread(img_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE | cv2.IMREAD_ANYDEPTH)
        mask = mask.astype(np.uint8)
        return image, mask

    def process_dataset(self, images: List[str], masks: List[str],
                        output_dir: Path, augment=True, max_samples=None):
        """Process a dataset split with optional augmentation"""
        pipeline = self.get_augmentation_pipeline(augment)
        total_original = len(images)
        pbar = tqdm(zip(images, masks), total=total_original, desc=f"Processing {output_dir.name}")
        saved_count = 0

        for img_path, mask_path in pbar:
            image, mask = self.load_image_mask_pair(img_path, mask_path)
            name = Path(img_path).stem

            if augment:
                for idx in range(self.n_augmentations):
                    transformed = pipeline(image=image, mask=mask)
                    self._save_pair(transformed["image"], transformed["mask"], name, idx, output_dir)
                    saved_count += 1
            else:
                transformed = pipeline(image=image, mask=mask)
                self._save_pair(transformed["image"], transformed["mask"], name, 0, output_dir)
                saved_count += 1

            pbar.set_postfix(saved=saved_count)
            if max_samples and saved_count >= max_samples:
                break

        print(f"{output_dir.name} set — {saved_count} samples saved.")

    def _save_pair(self, image, mask, name, idx, output_dir):
        """Save an image-mask pair as PNG"""
        suffix = f"_{idx:02d}" if idx > 0 else ""
        image_name = f"{name}{suffix}.png"
        mask_name = f"{name}{suffix}.png"

        image = (image * 255).astype(np.uint8) if image.max() <= 1.0 else image.astype(np.uint8)
        mask = mask.astype(np.uint8)

        cv2.imwrite(str(output_dir / "images" / image_name), cv2.cvtColor(image, cv2.COLOR_RGB2BGR))
        cv2.imwrite(str(output_dir / "masks" / mask_name), mask)


def main():
    """Command-line entry point"""
    parser = argparse.ArgumentParser(description="DRIVE Dataset Preprocessor with Flexible Augmentation and Resizing")
    parser.add_argument("--input", "-i", required=True, help="Path to the DRIVE dataset folder")
    parser.add_argument("--output", "-o", default="processed_data", help="Output directory path")
    parser.add_argument("--augmentations", "-a", type=int, default=4,
                        help="Number of augmentations per image (default: 4)")
    parser.add_argument("--resize", "-r", nargs=2, type=int, default=[512, 512],
                        help="Resize dimensions: width height (default: 512 512)")

    args = parser.parse_args()
    resize_dim = tuple(args.resize)

    np.random.seed(42)
    preprocessor = DRIVEPreprocessor(
        data_path=args.input,
        output_path=args.output,
        n_augmentations=args.augmentations,
        resize_dim=resize_dim
    )

    preprocessor.create_directories()
    paths = preprocessor.load_dataset_paths()

    preprocessor.process_dataset(paths['train_images'], paths['train_masks'],
                                 preprocessor.output_path / "train", augment=True)
    preprocessor.process_dataset(paths['val_images'], paths['val_masks'],
                                 preprocessor.output_path / "val", augment=False)
    preprocessor.process_dataset(paths['test_images'], paths['test_masks'],
                                 preprocessor.output_path / "test", augment=False)

    print("Preprocessing complete. All outputs are saved and ready for model training.")


if __name__ == "__main__":
    main()




