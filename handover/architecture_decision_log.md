# Architecture Decision Log (ADL)

## ADR-001: Baseline Model Selection, Data Ingestion, and Partitioning Strategy

- **Project:** Bright Future School Image Classification (Classroom vs. Playground)
- **Status:** Accepted
- **Author:** Mushtaq Ahmad
- **Task Scope:** Task 1 – Repository Setup, Data Pipeline, and Architecture Baseline

---

### 1. Context & Business Problem
Bright Future School requires an automated computer vision solution to organize and tag photographs captured during school events into **classroom** and **playground** categories. 
- **Operational Goal:** Help school staff swiftly sort visual archives for newsletters, publications, and social media.
- **Accuracy Target:** Minimum **70% classification accuracy** on unseen test data with low inference latency for web deployment.

---

### 2. Decision Summary

#### 2.1 Dataset Ingestion & Proxy Baseline
* **Dataset Choice:** `torchvision.datasets.CIFAR10`
* **Decision Rationale:** 
  - Standardized, programmatic ingestion ensures that the entire repository setup and CI/CD pipelines can be verified autonomously without manual data download gates.
  - Serves as the validated benchmark proxy to prove pipeline stability before domain adaptation to the custom school event dataset.
* **Storage Pattern:** 
  - Raw archives stored in `data/raw/` (git-ignored).
  - Unpacked into a standard directory-per-class hierarchy compatible with PyTorch's `ImageFolder`.

#### 2.2 Data Partitioning & Reproducibility
* **Splitting Strategy:**
  - **Training Partition (40,000 images / 80%):** Dedicated for model parameter optimization.
  - **Validation Partition (10,000 images / 20%):** Dedicated for hyperparameter tuning, model checkpoint selection, and early stopping.
  - **Test Partition (10,000 images):** Unbiased final evaluation set (standard CIFAR-10 test partition).
* **Determinism:** 
  - Splitting is governed by `torch.Generator().manual_seed(42)` to ensure identical partitions across distributed development machines and container builds.

#### 2.3 Model Architecture Selection (Task 2 Baseline)
* **Architecture Choice:** Pretrained **ResNet-18** (ImageNet-1k weights).
* **Decision Rationale:**
  - **Residual Connections:** Skip connections effectively counter vanishing gradient degradation, enabling smooth feature extraction.
  - **Transfer Learning Advantage:** Pre-trained ImageNet weights provide robust low-level filters (edges, textures, color gradients) that drastically accelerate convergence compared to training from scratch.
  - **Resource Footprint:** ResNet-18 has ~11.7 million parameters, ensuring fast CPU inference and modest memory usage well within free-tier container limits on Render (Task 5).
* **Adaptation:** Replace the final 1,000-class fully connected classification head (`fc`) with an output layer tailored to the target task.

#### 2.4 Preprocessing & Augmentation Strategy
* **Input Resolution & Normalization:**
  - Resize/scale inputs to match backbone receptive field requirements (or maintain $32 \times 32$ with adapted first conv).
  - Apply standard ImageNet channel normalization:
    $$\mu = [0.485, 0.456, 0.406], \quad \sigma = [0.229, 0.224, 0.225]$$
* **Planned Augmentations (Task 3):**
  - `RandomHorizontalFlip(p=0.5)`
  - `RandomCrop` with reflective padding
  - `ColorJitter` (brightness, contrast) to enhance generalization across varied school lighting conditions.

---

### 3. Alternatives Considered & Trade-offs

| Alternative | Pros | Cons | Verdict |
| :--- | :--- | :--- | :--- |
| **Custom CNN from scratch** | Tailored architecture, zero licensing constraints | Requires massive dataset to prevent overfitting; slow convergence | **Rejected:** Does not leverage transfer learning benefits. |
| **ResNet-50 / DenseNet** | Higher representational capacity | Heavier memory footprint (~25M+ params), slower inference on low-cost CPUs | **Rejected for Baseline:** Overkill for $\ge 70\%$ target; increases deployment risk on free-tier hosting. |
| **Pretrained ResNet-18** | Fast training, lightweight (~45 MB weights), robust feature extraction, easy Grad-CAM compatibility | Fixed input topology | **Accepted:** Optimal trade-off between accuracy, training speed, and deployment feasibility. |

---

### 4. Consequences & Downstream Impact
- **Positive:** Standardized folder structure (`data/train`, `data/val`, `data/test`) allows seamless loading via `torchvision.datasets.ImageFolder` in Task 2.
- **Positive:** Deterministic splits prevent data leakage and ensure reproducible baseline metrics.
- **Operational:** Large image files are strictly excluded from Git tracking via `.gitignore` to prevent repository bloat.
