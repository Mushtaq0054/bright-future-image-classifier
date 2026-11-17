import os
import sys
import argparse
import logging
from pathlib import Path
from typing import Tuple, List, Optional

# Ensure project root is on sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms, models
from PIL import Image
import numpy as np
import matplotlib
matplotlib.use("Agg")  # Non-interactive backend for headless saving
import matplotlib.pyplot as plt

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)

DEFAULT_DATA_DIR = PROJECT_ROOT / "data"
DEFAULT_RESULTS_DIR = PROJECT_ROOT / "results"
DEFAULT_GRADCAM_DIR = DEFAULT_RESULTS_DIR / "gradcam"
DEFAULT_MODEL_PATH = PROJECT_ROOT / "models" / "improved_resnet18_best.pt"


class GradCAM:
    """
    Gradient-weighted Class Activation Mapping (Grad-CAM) implementation
    for PyTorch Convolutional Neural Networks.
    """
    def __init__(self, model: nn.Module, target_layer: nn.Module):
        self.model = model
        self.target_layer = target_layer
        self.activations: Optional[torch.Tensor] = None
        self.gradients: Optional[torch.Tensor] = None

        # Register forward and backward hooks on target layer (layer4)
        self.forward_handle = self.target_layer.register_forward_hook(self._forward_hook)
        self.backward_handle = self.target_layer.register_full_backward_hook(self._backward_hook)

    def _forward_hook(self, module, input, output):
        self.activations = output.detach()

    def _backward_hook(self, module, grad_input, grad_output):
        self.gradients = grad_output[0].detach()

    def generate_heatmap(self, input_tensor: torch.Tensor, class_idx: int) -> np.ndarray:
        """
        Computes Grad-CAM heatmap for a given input tensor and target class index:
        alpha_k^c = (1/Z) * sum_i sum_j (d y^c / d A_ij^k)
        L_GradCAM^c = ReLU(sum_k alpha_k^c * A^k)
        """
        self.model.eval()
        self.model.zero_grad()

        # Forward pass
        output = self.model(input_tensor)
        target_score = output[0, class_idx]

        # Backward pass
        target_score.backward()

        gradients = self.gradients[0]     # [C, H, W]
        activations = self.activations[0] # [C, H, W]

        # Global Average Pooling of gradients to compute importance weights alpha
        alpha = torch.mean(gradients, dim=(1, 2), keepdim=True) # [C, 1, 1]

        # Linear combination of feature maps weighted by alpha
        weighted_combination = torch.sum(alpha * activations, dim=0) # [H, W]

        # Apply ReLU to keep only features having a positive influence on target class
        heatmap = F.relu(weighted_combination)

        # Normalize heatmap to [0, 1]
        heatmap = heatmap.cpu().numpy()
        min_val = np.min(heatmap)
        max_val = np.max(heatmap)
        if max_val - min_val > 1e-8:
            heatmap = (heatmap - min_val) / (max_val - min_val)
        else:
            heatmap = np.zeros_like(heatmap)

        return heatmap

    def remove_hooks(self):
        """Removes forward and backward hooks from model."""
        self.forward_handle.remove()
        self.backward_handle.remove()


def overlay_heatmap_on_image(
    image: Image.Image,
    heatmap: np.ndarray,
    alpha: float = 0.5,
    colormap_name: str = "jet",
) -> np.ndarray:
    """
    Upsamples heatmap to image dimensions, applies colormap, and blends with original image.
    """
    img_array = np.array(image.convert("RGB")).astype(np.float32) / 255.0
    h, w, _ = img_array.shape

    # Resize heatmap to match original image dimensions using PIL bicubic
    heatmap_pil = Image.fromarray((heatmap * 255).astype(np.uint8)).resize((w, h), resample=Image.BICUBIC)
    heatmap_resized = np.array(heatmap_pil).astype(np.float32) / 255.0

    # Apply colormap
    cmap = matplotlib.colormaps[colormap_name]
    colored_heatmap = cmap(heatmap_resized)[:, :, :3]  # Strip alpha channel

    # Blend
    blended = alpha * colored_heatmap + (1 - alpha) * img_array
    blended = np.clip(blended, 0.0, 1.0)
    return (blended * 255).astype(np.uint8)


def create_gradcam_figure(
    original_img: Image.Image,
    heatmap: np.ndarray,
    overlaid_img: np.ndarray,
    true_label: str,
    pred_label: str,
    confidence: float,
    output_path: Path,
):
    """
    Renders a 3-panel diagnostic visualization:
    [Original Image] | [Grad-CAM Heatmap] | [Overlay Visualization]
    """
    fig, axes = plt.subplots(1, 3, figsize=(12, 4), dpi=150)

    # 1. Original Image
    axes[0].imshow(original_img)
    axes[0].set_title(f"True Label: {true_label.upper()}", fontsize=12, fontweight="bold", color="#1a5276")
    axes[0].axis("off")

    # 2. Grad-CAM Heatmap
    im_heat = axes[1].imshow(heatmap, cmap="jet", vmin=0, vmax=1)
    axes[1].set_title("Grad-CAM Activation Map (layer4)", fontsize=11, fontweight="bold", color="#117a65")
    axes[1].axis("off")
    fig.colorbar(im_heat, ax=axes[1], fraction=0.046, pad=0.04)

    # 3. Overlaid Heatmap
    axes[2].imshow(overlaid_img)
    pred_color = "#b03a2e"  # Red for misclassification
    axes[2].set_title(
        f"Predicted: {pred_label.upper()} ({confidence:.1f}%)\n[MISCLASSIFIED]",
        fontsize=11,
        fontweight="bold",
        color=pred_color,
    )
    axes[2].axis("off")

    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(str(output_path), bbox_inches="tight")
    plt.close(fig)
    logger.info("Saved Grad-CAM diagnostic figure to: %s", output_path)


def generate_gradcam_for_samples(
    samples_list: List[dict],
    model: nn.Module,
    output_dir: Path,
    device: torch.device,
    class_names: List[str],
) -> List[Path]:
    """Processes a curated list of samples and generates Grad-CAM visualizations."""
    output_dir.mkdir(parents=True, exist_ok=True)
    gradcam = GradCAM(model=model, target_layer=model.layer4)

    eval_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
        ),
    ])

    saved_figures = []

    for idx, sample in enumerate(samples_list, start=1):
        rel_path = sample["image_path"]
        abs_img_path = PROJECT_ROOT / rel_path
        true_label = sample["true_label"]
        pred_label = sample["predicted_label"]
        conf = float(sample["predicted_confidence"])
        pred_label_idx = class_names.index(pred_label)

        if not abs_img_path.exists():
            logger.warning("Image path not found: %s", abs_img_path)
            continue

        orig_pil = Image.open(abs_img_path).convert("RGB")
        input_tensor = eval_transform(orig_pil).unsqueeze(0).to(device)

        # Generate Grad-CAM for the predicted class (shows what fooled the model)
        heatmap = gradcam.generate_heatmap(input_tensor=input_tensor, class_idx=pred_label_idx)

        # Overlay on original image
        overlaid = overlay_heatmap_on_image(orig_pil, heatmap, alpha=0.5)

        # Save figure
        fig_name = f"gradcam_error_{idx:02d}_{true_label}_as_{pred_label}.png"
        fig_path = output_dir / fig_name
        create_gradcam_figure(
            original_img=orig_pil,
            heatmap=heatmap,
            overlaid_img=overlaid,
            true_label=true_label,
            pred_label=pred_label,
            confidence=conf,
            output_path=fig_path,
        )
        saved_figures.append(fig_path)

    gradcam.remove_hooks()
    return saved_figures


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate Grad-CAM visual heatmaps for error analysis.")
    parser.add_argument("--model-path", type=str, default=str(DEFAULT_MODEL_PATH), help="Model weights path.")
    parser.add_argument("--output-dir", type=str, default=str(DEFAULT_GRADCAM_DIR), help="Output folder.")
    parser.add_argument("--csv-path", type=str, default=str(DEFAULT_RESULTS_DIR / "misclassified_samples.csv"), help="CSV list.")
    parser.add_argument("--count", type=int, default=5, help="Number of representative errors to generate.")
    args = parser.parse_args()

    import csv
    from src.error_analysis import load_model_for_evaluation

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    class_names = ["airplane", "automobile", "bird", "cat", "deer", "dog", "frog", "horse", "ship", "truck"]
    model = load_model_for_evaluation(Path(args.model_path), num_classes=len(class_names), device=device)

    # Load top representative misclassified samples from CSV
    samples = []
    with open(args.csv_path, mode="r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for i, row in enumerate(reader):
            if i >= args.count:
                break
            samples.append(row)

    generate_gradcam_for_samples(
        samples_list=samples,
        model=model,
        output_dir=Path(args.output_dir),
        device=device,
        class_names=class_names,
    )
