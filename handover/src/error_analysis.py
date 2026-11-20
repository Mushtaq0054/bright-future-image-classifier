import os
import sys
import csv
import argparse
import logging
from pathlib import Path
from typing import Dict, List, Tuple, Any

# Ensure project root is on sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms, models
from tqdm import tqdm

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)

DEFAULT_DATA_DIR = PROJECT_ROOT / "data"
DEFAULT_RESULTS_DIR = PROJECT_ROOT / "results"
DEFAULT_MODEL_PATH = PROJECT_ROOT / "models" / "improved_resnet18_best.pt"


def load_model_for_evaluation(model_path: Path, num_classes: int = 10, device: torch.device = None) -> nn.Module:
    """Loads ResNet-18 model architecture and restores trained weights."""
    if not model_path.exists():
        raise FileNotFoundError(f"Model checkpoint not found at {model_path}.")

    model = models.resnet18(weights=None)
    model.fc = nn.Linear(model.fc.in_features, num_classes)

    state_dict = torch.load(str(model_path), map_location=device, weights_only=True)
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()
    logger.info("Loaded model weights from %s on device %s", model_path, device)
    return model


def get_test_dataset(data_dir: Path) -> Tuple[datasets.ImageFolder, DataLoader]:
    """Prepares test dataset and DataLoader with standard evaluation normalization."""
    test_dir = data_dir / "test"
    if not test_dir.exists():
        raise FileNotFoundError(f"Test directory not found at {test_dir}. Run data_pipeline.py first.")

    eval_transforms = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
        ),
    ])

    test_dataset = datasets.ImageFolder(root=str(test_dir), transform=eval_transforms)
    test_loader = DataLoader(
        test_dataset,
        batch_size=128,
        shuffle=False,
        num_workers=0,
        pin_memory=torch.cuda.is_available(),
    )
    return test_dataset, test_loader


def run_error_analysis(
    data_dir: Path = DEFAULT_DATA_DIR,
    results_dir: Path = DEFAULT_RESULTS_DIR,
    model_path: Path = DEFAULT_MODEL_PATH,
    output_csv_name: str = "misclassified_samples.csv",
) -> List[Dict[str, Any]]:
    """
    Evaluates the model on the test partition, identifies misclassified images,
    and outputs a structured CSV with full diagnostic details.
    """
    results_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if not torch.cuda.is_available():
        torch.set_num_threads(min(8, os.cpu_count() or 4))

    test_dataset, test_loader = get_test_dataset(data_dir)
    class_names = test_dataset.classes
    model = load_model_for_evaluation(model_path, num_classes=len(class_names), device=device)

    logger.info("Evaluating on %d test images across %d classes...", len(test_dataset), len(class_names))

    total_samples = 0
    correct_samples = 0
    misclassified_list = []

    # Get raw image file paths from ImageFolder samples: list of (path, label)
    sample_paths = [s[0] for s in test_dataset.samples]

    sample_idx = 0
    with torch.no_grad():
        for images, labels in tqdm(test_loader, desc="Testing"):
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            probabilities = torch.softmax(outputs, dim=1)

            # Top-2 predictions to check second-highest confidence
            top2_probs, top2_indices = torch.topk(probabilities, k=2, dim=1)

            batch_size = labels.size(0)
            for i in range(batch_size):
                global_idx = sample_idx + i
                true_label_idx = labels[i].item()
                pred_label_idx = top2_indices[i, 0].item()
                pred_conf = top2_probs[i, 0].item() * 100.0

                second_label_idx = top2_indices[i, 1].item()
                second_conf = top2_probs[i, 1].item() * 100.0

                total_samples += 1
                if pred_label_idx == true_label_idx:
                    correct_samples += 1
                else:
                    rel_path = os.path.relpath(sample_paths[global_idx], str(PROJECT_ROOT))
                    misclassified_list.append({
                        "sample_idx": global_idx,
                        "image_path": rel_path.replace("\\", "/"),
                        "true_label": class_names[true_label_idx],
                        "true_label_idx": true_label_idx,
                        "predicted_label": class_names[pred_label_idx],
                        "predicted_label_idx": pred_label_idx,
                        "predicted_confidence": round(pred_conf, 2),
                        "second_predicted_label": class_names[second_label_idx],
                        "second_confidence": round(second_conf, 2),
                    })

            sample_idx += batch_size

    test_accuracy = (correct_samples / total_samples) * 100.0
    logger.info("=== Test Evaluation Complete ===")
    logger.info("Total Test Samples: %d", total_samples)
    logger.info("Correctly Classified: %d", correct_samples)
    logger.info("Misclassified Samples: %d", len(misclassified_list))
    logger.info("Overall Test Accuracy: %.2f%%", test_accuracy)

    # Sort misclassifications by confidence descending (high-confidence errors first)
    misclassified_list.sort(key=lambda x: x["predicted_confidence"], reverse=True)

    # Save to CSV
    csv_file = results_dir / output_csv_name
    with open(csv_file, mode="w", newline="", encoding="utf-8") as f:
        fieldnames = [
            "sample_idx",
            "image_path",
            "true_label",
            "predicted_label",
            "predicted_confidence",
            "second_predicted_label",
            "second_confidence",
        ]
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(misclassified_list)

    logger.info("Misclassified samples catalog saved to: %s", csv_file)
    return misclassified_list


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Perform error analysis on the test set.")
    parser.add_argument("--data-dir", type=str, default=str(DEFAULT_DATA_DIR), help="Data directory.")
    parser.add_argument("--results-dir", type=str, default=str(DEFAULT_RESULTS_DIR), help="Results directory.")
    parser.add_argument("--model-path", type=str, default=str(DEFAULT_MODEL_PATH), help="Path to trained model.")
    parser.add_argument("--output-csv", type=str, default="misclassified_samples.csv", help="CSV filename.")
    args = parser.parse_args()

    run_error_analysis(
        data_dir=Path(args.data_dir),
        results_dir=Path(args.results_dir),
        model_path=Path(args.model_path),
        output_csv_name=args.output_csv,
    )
