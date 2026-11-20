import os
import sys
import time
import csv
import argparse
import logging
from pathlib import Path
from typing import Tuple, List

# Ensure project root is on sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms, models
from tqdm import tqdm

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)

# Dynamic paths relative to project root
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_DATA_DIR = PROJECT_ROOT / "data"
DEFAULT_RESULTS_DIR = PROJECT_ROOT / "results"
DEFAULT_MODELS_DIR = PROJECT_ROOT / "models"


def get_improved_transforms() -> Tuple[transforms.Compose, transforms.Compose]:
    """
    Returns data augmentation transforms for training and standard normalization for evaluation.
    Augmentation includes RandomCrop, RandomHorizontalFlip, and ColorJitter.
    """
    train_transforms = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
        ),
    ])

    val_transforms = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
        ),
    ])

    return train_transforms, val_transforms


def compute_class_weights(dataset: datasets.ImageFolder) -> torch.Tensor:
    """
    Computes inverse class frequencies as weights to penalize minority class errors in loss function:
    w_c = Total_Samples / (Num_Classes * Count_c)
    """
    # Extract class labels for all samples
    targets = [sample[1] for sample in dataset.samples]
    class_counts = torch.bincount(torch.tensor(targets, dtype=torch.long))
    total_samples = len(targets)
    num_classes = len(class_counts)

    class_weights = total_samples / (num_classes * class_counts.float())
    logger.info("Computed class weights for %d classes:", num_classes)
    for idx, (class_name, count, weight) in enumerate(zip(dataset.classes, class_counts, class_weights)):
        logger.info("  Class %d (%s): count=%d, weight=%.4f", idx, class_name, count.item(), weight.item())

    return class_weights


def get_improved_data_loaders(
    data_dir: Path,
    batch_size: int = 64,
    num_workers: int = 0,
    subset_size: int = None,
    seed: int = 42,
) -> Tuple[DataLoader, DataLoader, List[str], torch.Tensor]:
    """
    Builds augmented train loader, validation loader, class names, and class weights.
    Optionally samples a deterministic stratified subset for fast hyperparameter search.
    """
    train_dir = data_dir / "train"
    val_dir = data_dir / "val"

    if not train_dir.exists() or not val_dir.exists():
        raise FileNotFoundError(
            f"Train/Val directories not found in {data_dir}. "
            "Please run 'python src/data_pipeline.py' first."
        )

    train_transforms, val_transforms = get_improved_transforms()

    train_dataset = datasets.ImageFolder(root=str(train_dir), transform=train_transforms)
    val_dataset = datasets.ImageFolder(root=str(val_dir), transform=val_transforms)

    class_weights = compute_class_weights(train_dataset)

    # Optional subset for fast trials
    if subset_size is not None and subset_size < len(train_dataset):
        torch.manual_seed(seed)
        indices = torch.randperm(len(train_dataset))[:subset_size].tolist()
        train_dataset = Subset(train_dataset, indices)
        logger.info("Using subset of %d training samples for search trial.", subset_size)

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
    )

    return train_loader, val_loader, val_dataset.classes, class_weights


def build_improved_resnet18(num_classes: int = 10, pretrained: bool = True, init_checkpoint: Path = None) -> nn.Module:
    """Loads ResNet-18 pre-trained on ImageNet and adapts classification head, optionally loading existing weights."""
    weights = models.ResNet18_Weights.DEFAULT if pretrained else None
    model = models.resnet18(weights=weights)
    in_features = model.fc.in_features
    model.fc = nn.Linear(in_features, num_classes)

    if init_checkpoint is not None and init_checkpoint.exists():
        logger.info("Initializing improved model from baseline checkpoint: %s", init_checkpoint)
        state_dict = torch.load(str(init_checkpoint), map_location="cpu", weights_only=True)
        model.load_state_dict(state_dict)
        logger.info("Loaded baseline checkpoint weights successfully.")

    return model


def train_one_epoch(model, dataloader, criterion, optimizer, device):
    """Executes a single training epoch with augmentation and weighted loss."""
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    pbar = tqdm(dataloader, desc="Training Batch", leave=False)
    for images, labels in pbar:
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * images.size(0)
        _, preds = torch.max(outputs, 1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)

        pbar.set_postfix(
            batch_loss=f"{loss.item():.4f}",
            batch_acc=f"{(correct / total) * 100:.1f}%",
        )

    epoch_loss = running_loss / total
    epoch_acc = (correct / total) * 100.0
    return epoch_loss, epoch_acc


def evaluate(model, dataloader, criterion, device):
    """Evaluates the model on validation data."""
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for images, labels in tqdm(dataloader, desc="Evaluating", leave=False):
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)

            running_loss += loss.item() * images.size(0)
            _, preds = torch.max(outputs, 1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)

    epoch_loss = running_loss / total
    epoch_acc = (correct / total) * 100.0
    return epoch_loss, epoch_acc


def train_improved_model(
    data_dir: Path = DEFAULT_DATA_DIR,
    results_dir: Path = DEFAULT_RESULTS_DIR,
    models_dir: Path = DEFAULT_MODELS_DIR,
    epochs: int = 10,
    batch_size: int = 64,
    learning_rate: float = 0.0003,
    num_workers: int = 0,
    subset_size: int = None,
    save_name: str = "improved_resnet18_best.pt",
    csv_log_name: str = "improved_training_metrics.csv",
    init_checkpoint: Path = None,
) -> float:
    """Orchestrates end-to-end training of the improved model with augmentation & class weights."""
    results_dir.mkdir(parents=True, exist_ok=True)
    models_dir.mkdir(parents=True, exist_ok=True)

    if not torch.cuda.is_available():
        num_threads = min(8, os.cpu_count() or 4)
        torch.set_num_threads(num_threads)
        logger.info("Configured PyTorch CPU threads: %d", num_threads)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info("Using compute device: %s", device)

    # 1. Load data with augmentations and class weights
    train_loader, val_loader, class_names, class_weights = get_improved_data_loaders(
        data_dir=data_dir,
        batch_size=batch_size,
        num_workers=num_workers,
        subset_size=subset_size,
    )

    # 2. Build model
    model = build_improved_resnet18(
        num_classes=len(class_names),
        pretrained=True,
        init_checkpoint=init_checkpoint,
    )
    model = model.to(device)

    # 3. Weighted CrossEntropyLoss + Adam optimizer
    criterion = nn.CrossEntropyLoss(weight=class_weights.to(device))
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    # 4. Prepare CSV logging
    csv_path = results_dir / csv_log_name
    with open(csv_path, mode="w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["epoch", "train_loss", "train_acc", "val_loss", "val_acc", "time_seconds"])

    best_val_acc = 0.0
    best_model_path = models_dir / save_name

    logger.info(
        "Training improved model with Augmentation + Class Weights (lr=%.6f, batch_size=%d) for %d epochs...",
        learning_rate,
        batch_size,
        epochs,
    )
    total_start = time.time()

    for epoch in range(1, epochs + 1):
        t0 = time.time()
        train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_acc = evaluate(model, val_loader, criterion, device)
        duration = time.time() - t0

        with open(csv_path, mode="a", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow([epoch, f"{train_loss:.4f}", f"{train_acc:.2f}", f"{val_loss:.4f}", f"{val_acc:.2f}", f"{duration:.1f}"])

        is_best = val_acc > best_val_acc
        if is_best:
            best_val_acc = val_acc
            torch.save(model.state_dict(), str(best_model_path))

        star = "★ (New Best)" if is_best else ""
        logger.info(
            "Epoch [%02d/%02d] | Train Loss: %.4f | Train Acc: %.2f%% | Val Loss: %.4f | Val Acc: %.2f%% | Time: %.1fs %s",
            epoch, epochs, train_loss, train_acc, val_loss, val_acc, duration, star
        )

    logger.info("Training complete in %.1f minutes. Best Val Accuracy: %.2f%%", (time.time() - total_start) / 60, best_val_acc)
    logger.info("Model saved to: %s", best_model_path)
    logger.info("Metrics logged to: %s", csv_path)
    return best_val_acc


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train improved ResNet-18 with augmentation and class weights.")
    parser.add_argument("--epochs", type=int, default=10, help="Training epochs (default: 10).")
    parser.add_argument("--batch-size", type=int, default=64, help="Batch size (default: 64).")
    parser.add_argument("--lr", type=float, default=0.0003, help="Learning rate (default: 0.0003).")
    parser.add_argument("--num-workers", type=int, default=0, help="DataLoader workers (default: 0).")
    parser.add_argument("--data-dir", type=str, default=str(DEFAULT_DATA_DIR), help="Data directory path.")
    parser.add_argument("--results-dir", type=str, default=str(DEFAULT_RESULTS_DIR), help="Results directory path.")
    parser.add_argument("--models-dir", type=str, default=str(DEFAULT_MODELS_DIR), help="Models directory path.")
    parser.add_argument("--init-checkpoint", type=str, default=None, help="Optional checkpoint path to initialize weights.")
    args = parser.parse_args()

    train_improved_model(
        data_dir=Path(args.data_dir),
        results_dir=Path(args.results_dir),
        models_dir=Path(args.models_dir),
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.lr,
        num_workers=args.num_workers,
        init_checkpoint=Path(args.init_checkpoint) if args.init_checkpoint else None,
    )
