import os
import argparse
import logging
from pathlib import Path
import torch
from torchvision import datasets
from torch.utils.data import random_split
from PIL import Image
from tqdm import tqdm

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)

# Determine project root dynamically relative to this script
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_DATA_ROOT = PROJECT_ROOT / "data"

def is_dataset_already_prepared(train_dir: Path, val_dir: Path, test_dir: Path) -> bool:
    """Check if train, val, and test directories already contain exported images."""
    for folder in [train_dir, val_dir, test_dir]:
        if not folder.exists():
            return False
        # Check if folder has subdirectories and at least one image
        has_images = any(folder.glob("*/*.png"))
        if not has_images:
            return False
    return True

def prepare_cifar10_data(data_root=DEFAULT_DATA_ROOT, val_ratio=0.2, seed=42, force=False):
    """
    Downloads CIFAR-10, splits it into train/val/test sets, and saves them
    into PyTorch ImageFolder compatible directories.
    """
    data_root = Path(data_root)
    raw_dir = data_root / "raw"
    train_dir = data_root / "train"
    val_dir = data_root / "val"
    test_dir = data_root / "test"

    # Idempotency check: avoid redundant processing if data already exists
    if not force and is_dataset_already_prepared(train_dir, val_dir, test_dir):
        logger.info("Dataset partitions already exist and contain images in %s.", data_root)
        logger.info("Skipping data export. Use '--force' flag to re-download and re-export.")
        return

    # Ensure required directories exist
    for d in [raw_dir, train_dir, val_dir, test_dir]:
        d.mkdir(parents=True, exist_ok=True)

    logger.info("Downloading CIFAR-10 dataset via torchvision...")
    full_trainset = datasets.CIFAR10(root=str(raw_dir), train=True, download=True)
    testset = datasets.CIFAR10(root=str(raw_dir), train=False, download=True)

    # Train / Validation split
    total_train = len(full_trainset)
    val_size = int(val_ratio * total_train)
    train_size = total_train - val_size

    generator = torch.Generator().manual_seed(seed)
    train_subset, val_subset = random_split(
        full_trainset, 
        [train_size, val_size],
        generator=generator
    )

    class_names = full_trainset.classes
    logger.info("Detected %d classes: %s", len(class_names), ", ".join(class_names))

    def export_images(dataset_subset, target_folder: Path, subset_name: str):
        for class_name in class_names:
            (target_folder / class_name).mkdir(parents=True, exist_ok=True)

        logger.info("Exporting %s images (%d total) to %s...", subset_name, len(dataset_subset), target_folder)
        for idx in tqdm(range(len(dataset_subset)), desc=f"Exporting {subset_name}"):
            img, label = dataset_subset[idx]
            class_name = class_names[label]
            file_path = target_folder / class_name / f"img_{idx:05d}.png"
            img.save(file_path)

    export_images(train_subset, train_dir, "train")
    export_images(val_subset, val_dir, "val")
    export_images(testset, test_dir, "test")

    logger.info("Dataset preparation complete:")
    logger.info("- Train set: %d images in %s", len(train_subset), train_dir)
    logger.info("- Val set:   %d images in %s", len(val_subset), val_dir)
    logger.info("- Test set:  %d images in %s", len(testset), test_dir)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Download and prepare CIFAR-10 dataset.")
    parser.add_argument(
        "--data-root",
        type=str,
        default=str(DEFAULT_DATA_ROOT),
        help="Root directory for dataset storage."
    )
    parser.add_argument(
        "--val-ratio",
        type=float,
        default=0.2,
        help="Validation split ratio (default: 0.2)."
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducible train/val split (default: 42)."
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Force re-export even if data directories already exist."
    )
    args = parser.parse_args()

    prepare_cifar10_data(
        data_root=args.data_root,
        val_ratio=args.val_ratio,
        seed=args.seed,
        force=args.force
    )
