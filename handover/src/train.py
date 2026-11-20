import os
import time
import csv
import argparse
import logging
from pathlib import Path
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
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


def get_data_loaders(data_dir: Path, batch_size: int = 128, num_workers: int = 0):
    """
    Sets up DataLoaders for train and validation splits with ImageNet normalization.
    """
    train_dir = data_dir / "train"
    val_dir = data_dir / "val"

    if not train_dir.exists() or not val_dir.exists():
        raise FileNotFoundError(
            f"Train/Val directories not found in {data_dir}. "
            "Please run 'python src/data_pipeline.py' first."
        )

    # Standard normalization matching pre-trained ImageNet backbone
    transform_pipeline = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
        ),
    ])

    train_dataset = datasets.ImageFolder(root=str(train_dir), transform=transform_pipeline)
    val_dataset = datasets.ImageFolder(root=str(val_dir), transform=transform_pipeline)

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

    logger.info(
        "Loaded datasets: %d train images, %d validation images across %d classes.",
        len(train_dataset),
        len(val_dataset),
        len(train_dataset.classes),
    )
    return train_loader, val_loader, train_dataset.classes


def build_resnet18_baseline(num_classes: int = 10, pretrained: bool = True) -> nn.Module:
    """
    Loads ResNet-18 (pre-trained on ImageNet) and replaces the classification head
    for the target number of classes.
    """
    weights = models.ResNet18_Weights.DEFAULT if pretrained else None
    model = models.resnet18(weights=weights)

    # Replace final 1000-class FC layer with target classification head
    in_features = model.fc.in_features
    model.fc = nn.Linear(in_features, num_classes)
    logger.info(
        "Configured ResNet-18 with adapted classification head: Linear(in_features=%d, out_features=%d).",
        in_features,
        num_classes,
    )
    return model


def train_one_epoch(model, dataloader, criterion, optimizer, device):
    """Executes a single training epoch and returns average loss and accuracy."""
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
    """Evaluates the model on validation data and returns average loss and accuracy."""
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


def run_training(
    data_dir: Path = DEFAULT_DATA_DIR,
    results_dir: Path = DEFAULT_RESULTS_DIR,
    models_dir: Path = DEFAULT_MODELS_DIR,
    epochs: int = 10,
    batch_size: int = 128,
    learning_rate: float = 0.001,
    num_workers: int = 0,
):
    """Orchestrates end-to-end training and validation logging."""
    results_dir.mkdir(parents=True, exist_ok=True)
    models_dir.mkdir(parents=True, exist_ok=True)

    if not torch.cuda.is_available():
        num_threads = min(8, os.cpu_count() or 4)
        torch.set_num_threads(num_threads)
        logger.info("Configured PyTorch CPU threads: %d", num_threads)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info("Using compute device: %s", device)

    # 1. Load Data
    train_loader, val_loader, class_names = get_data_loaders(
        data_dir=data_dir,
        batch_size=batch_size,
        num_workers=num_workers,
    )

    # 2. Build Model
    model = build_resnet18_baseline(num_classes=len(class_names), pretrained=True)
    model = model.to(device)

    # 3. Loss & Optimizer (Adam optimizer as requested in task brief)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    # 4. Prepare CSV logging
    csv_path = results_dir / "training_metrics.csv"
    with open(csv_path, mode="w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["epoch", "train_loss", "train_acc", "val_loss", "val_acc", "time_seconds"])

    best_val_acc = 0.0
    best_model_path = models_dir / "baseline_resnet18_best.pt"

    logger.info("Starting baseline training for %d epochs...", epochs)
    total_start_time = time.time()

    for epoch in range(1, epochs + 1):
        epoch_start = time.time()

        train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_acc = evaluate(model, val_loader, criterion, device)

        epoch_duration = time.time() - epoch_start

        # Log to CSV
        with open(csv_path, mode="a", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow([
                epoch,
                f"{train_loss:.4f}",
                f"{train_acc:.2f}",
                f"{val_loss:.4f}",
                f"{val_acc:.2f}",
                f"{epoch_duration:.1f}",
            ])

        # Track and save best checkpoint
        is_best = val_acc > best_val_acc
        if is_best:
            best_val_acc = val_acc
            torch.save(model.state_dict(), str(best_model_path))

        status_marker = "★ (New Best)" if is_best else ""
        logger.info(
            "Epoch [%02d/%02d] | Train Loss: %.4f | Train Acc: %.2f%% | Val Loss: %.4f | Val Acc: %.2f%% | Time: %.1fs %s",
            epoch,
            epochs,
            train_loss,
            train_acc,
            val_loss,
            val_acc,
            epoch_duration,
            status_marker,
        )

    total_duration = time.time() - total_start_time
    logger.info("Training complete in %.1f minutes.", total_duration / 60)
    logger.info("Best Validation Accuracy Achieved: %.2f%%", best_val_acc)
    logger.info("Best model checkpoint saved to: %s", best_model_path)
    logger.info("Training metrics CSV logged to: %s", csv_path)

    # Verify Task 2 success threshold (70%)
    if best_val_acc >= 70.0:
        logger.info("SUCCESS: Validation accuracy exceeded target threshold of 70%%!")
    else:
        logger.warning("Target 70%% accuracy was not achieved. Consider training for additional epochs.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train ResNet-18 baseline model on CIFAR-10.")
    parser.add_argument("--epochs", type=int, default=10, help="Number of training epochs (default: 10).")
    parser.add_argument("--batch-size", type=int, default=128, help="Batch size for training and eval (default: 128).")
    parser.add_argument("--lr", type=float, default=0.001, help="Learning rate for Adam optimizer (default: 0.001).")
    parser.add_argument("--num-workers", type=int, default=0, help="Number of worker threads for DataLoaders (default: 0).")
    parser.add_argument("--data-dir", type=str, default=str(DEFAULT_DATA_DIR), help="Path to data directory.")
    parser.add_argument("--results-dir", type=str, default=str(DEFAULT_RESULTS_DIR), help="Path to results directory.")
    parser.add_argument("--models-dir", type=str, default=str(DEFAULT_MODELS_DIR), help="Path to save models.")

    args = parser.parse_args()

    run_training(
        data_dir=Path(args.data_dir),
        results_dir=Path(args.results_dir),
        models_dir=Path(args.models_dir),
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.lr,
        num_workers=args.num_workers,
    )
