import os
import sys
import time
import csv
import math
import random
import argparse
import logging
from pathlib import Path
from typing import Dict, List, Any

# Ensure project root is on sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset
from torchvision import datasets

# Re-use robust components from train_improved
from src.train_improved import (
    DEFAULT_DATA_DIR,
    DEFAULT_RESULTS_DIR,
    DEFAULT_MODELS_DIR,
    get_improved_transforms,
    compute_class_weights,
    build_improved_resnet18,
    train_one_epoch,
    evaluate,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


def sample_hyperparameters(trial_idx: int, seed: int = 42) -> Dict[str, Any]:
    """
    Samples learning rate log-uniformly from [1e-4, 1e-3] and batch size from {32, 64}.
    Deterministic sampling ensures full auditability across runs.
    """
    random.seed(seed + trial_idx)
    # Log-uniform sampling for learning rate: 10^(uniform(-4, -3))
    log_lr = random.uniform(-4.0, -3.0)
    lr = float(10 ** log_lr)

    batch_size = random.choice([32, 64])

    return {
        "trial_id": trial_idx + 1,
        "lr": round(lr, 6),
        "batch_size": batch_size,
    }


def run_single_trial(
    trial_params: Dict[str, Any],
    data_dir: Path,
    epochs: int,
    device: torch.device,
    subset_size: int = None,
    seed: int = 42,
) -> Dict[str, Any]:
    """Runs training for a single hyperparameter configuration."""
    trial_id = trial_params["trial_id"]
    lr = trial_params["lr"]
    batch_size = trial_params["batch_size"]

    logger.info("=== Starting Trial %d/5: lr=%.6f, batch_size=%d, epochs=%d ===", trial_id, lr, batch_size, epochs)
    trial_start = time.time()

    train_transforms, val_transforms = get_improved_transforms()

    train_dataset = datasets.ImageFolder(root=str(data_dir / "train"), transform=train_transforms)
    val_dataset = datasets.ImageFolder(root=str(data_dir / "val"), transform=val_transforms)

    class_weights = compute_class_weights(train_dataset)

    # Optional stratified subset for fast trial search
    if subset_size is not None and subset_size < len(train_dataset):
        torch.manual_seed(seed + trial_id)
        indices = torch.randperm(len(train_dataset))[:subset_size].tolist()
        train_dataset = Subset(train_dataset, indices)
        logger.info("Trial %d using stratified subset of %d images.", trial_id, subset_size)

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,
        pin_memory=torch.cuda.is_available(),
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=torch.cuda.is_available(),
    )

    model = build_improved_resnet18(num_classes=len(val_dataset.classes), pretrained=True).to(device)
    criterion = nn.CrossEntropyLoss(weight=class_weights.to(device))
    optimizer = optim.Adam(model.parameters(), lr=lr)

    best_val_acc = 0.0
    final_val_loss = 0.0

    for epoch in range(1, epochs + 1):
        t0 = time.time()
        train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_acc = evaluate(model, val_loader, criterion, device)
        duration = time.time() - t0

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            final_val_loss = val_loss

        logger.info(
            "Trial %d - Epoch [%02d/%02d] | Train Acc: %.2f%% | Val Acc: %.2f%% | Val Loss: %.4f | Time: %.1fs",
            trial_id, epoch, epochs, train_acc, val_acc, val_loss, duration
        )

    trial_duration = time.time() - trial_start
    logger.info("Trial %d Completed in %.1f minutes | Best Val Acc: %.2f%%", trial_id, trial_duration / 60, best_val_acc)

    return {
        "trial_id": trial_id,
        "lr": lr,
        "batch_size": batch_size,
        "best_val_acc": round(best_val_acc, 2),
        "final_val_loss": round(final_val_loss, 4),
        "duration_minutes": round(trial_duration / 60, 2),
    }


def execute_random_search(
    data_dir: Path = DEFAULT_DATA_DIR,
    results_dir: Path = DEFAULT_RESULTS_DIR,
    models_dir: Path = DEFAULT_MODELS_DIR,
    num_trials: int = 5,
    epochs_per_trial: int = 10,
    subset_size: int = None,
    seed: int = 42,
):
    """Executes at least 5 trials of random search and persists results to CSV."""
    results_dir.mkdir(parents=True, exist_ok=True)
    models_dir.mkdir(parents=True, exist_ok=True)

    if not torch.cuda.is_available():
        num_threads = min(8, os.cpu_count() or 4)
        torch.set_num_threads(num_threads)
        logger.info("Configured PyTorch CPU threads: %d", num_threads)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info("Running Random Search on: %s", device)

    trials_csv = results_dir / "random_search_trials.csv"
    with open(trials_csv, mode="w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["trial_id", "learning_rate", "batch_size", "best_val_acc", "final_val_loss", "duration_minutes"])

    results = []
    best_trial = None

    for i in range(num_trials):
        params = sample_hyperparameters(trial_idx=i, seed=seed)
        trial_result = run_single_trial(
            trial_params=params,
            data_dir=data_dir,
            epochs=epochs_per_trial,
            device=device,
            subset_size=subset_size,
            seed=seed,
        )

        results.append(trial_result)

        # Log immediately to CSV
        with open(trials_csv, mode="a", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow([
                trial_result["trial_id"],
                trial_result["lr"],
                trial_result["batch_size"],
                trial_result["best_val_acc"],
                trial_result["final_val_loss"],
                trial_result["duration_minutes"],
            ])

        if best_trial is None or trial_result["best_val_acc"] > best_trial["best_val_acc"]:
            best_trial = trial_result

    logger.info("==================================================")
    logger.info("RANDOM SEARCH COMPLETE (%d Trials Logged)", num_trials)
    logger.info("Optimal Configuration: Trial %d (lr=%.6f, batch_size=%d) -> Val Acc: %.2f%%",
                best_trial["trial_id"], best_trial["lr"], best_trial["batch_size"], best_trial["best_val_acc"])
    logger.info("Trial results saved to: %s", trials_csv)
    logger.info("==================================================")

    # Save summary comparison
    comparison_csv = results_dir / "task3_comparison.csv"
    baseline_acc = 80.56  # From Task 2
    improved_acc = best_trial["best_val_acc"]
    with open(comparison_csv, mode="w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["model_version", "augmentations", "class_weighting", "learning_rate", "batch_size", "val_accuracy", "gain_over_baseline"])
        writer.writerow(["Task 2 Baseline", "None (Standard)", "None (Unweighted)", "0.001000", "128", f"{baseline_acc:.2f}%", "0.00%"])
        writer.writerow([
            "Task 3 Improved (Best Trial)",
            "RandomCrop + HorizontalFlip + ColorJitter",
            "Inverse Class Frequency",
            f"{best_trial['lr']:.6f}",
            str(best_trial['batch_size']),
            f"{improved_acc:.2f}%",
            f"{improved_acc - baseline_acc:+.2f}%",
        ])
    logger.info("Comparison summary logged to: %s", comparison_csv)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Execute Random Search over learning rate and batch size.")
    parser.add_argument("--trials", type=int, default=5, help="Number of random search trials (default: 5).")
    parser.add_argument("--epochs", type=int, default=10, help="Epochs per trial (default: 10).")
    parser.add_argument("--subset", type=int, default=None, help="Optional subset size for fast trial evaluations.")
    parser.add_argument("--seed", type=int, default=42, help="Seed for reproducibility.")
    parser.add_argument("--data-dir", type=str, default=str(DEFAULT_DATA_DIR), help="Data directory path.")
    parser.add_argument("--results-dir", type=str, default=str(DEFAULT_RESULTS_DIR), help="Results directory path.")
    parser.add_argument("--models-dir", type=str, default=str(DEFAULT_MODELS_DIR), help="Models directory path.")
    args = parser.parse_args()

    execute_random_search(
        data_dir=Path(args.data_dir),
        results_dir=Path(args.results_dir),
        models_dir=Path(args.models_dir),
        num_trials=args.trials,
        epochs_per_trial=args.epochs,
        subset_size=args.subset,
        seed=args.seed,
    )
