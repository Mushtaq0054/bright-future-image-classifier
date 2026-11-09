# Bright Future School: Classroom vs Playground Image Classifier

Automated computer vision classification system developed for **Bright Future School**. The system automatically categorizes event photos into **classroom** and **playground** scenes, streamlining digital media organization for school newsletters, archives, and social media.

---

## Project Overview
- **Client:** Bright Future School
- **Objective:** Build, evaluate, and deploy an image classification pipeline achieving at least **70% accuracy**.
- **Current Phase:** Task 2 – Build and Evaluate a Transfer-Learning Baseline Model (Completed).

---

## Task Deliverables Summary

### Task 1: Repository Setup & Data Pipeline
- Modular, production-ready directory layout (`src/`, `data/`, `docs/`, `results/`).
- Automated data ingestion and 3-way split pipeline script (`train/`, `val/`, `test/`).
- Architecture Decision Log (ADL) documenting model selection, split rationale, and preprocessing strategy.
- Git configuration with comprehensive `.gitignore` for ML artifacts and datasets.

### Task 2: Transfer Learning Baseline Model
- Pre-trained **ResNet-18** backbone loaded from PyTorch Torchvision with ImageNet-1k weights.
- Final fully-connected classification head adapted for 10 target classes (`Linear(512, 10)`).
- End-to-end training pipeline (`src/train.py`) utilizing Adam optimizer (`lr=0.001`) and CrossEntropyLoss.
- Trained for 10 full epochs with comprehensive metric logging to `results/training_metrics.csv`.
- Achieved **80.56% Best Validation Accuracy**, comfortably surpassing the mentor requirement of $\ge 70\%$.
- Serialized best model weights saved locally at `models/baseline_resnet18_best.pt`.

---

## Project Directory Structure

```text
bright-future-image-classifier/
├── data/                         # Ignored by git
│   ├── raw/                      # Ingested archive files
│   ├── train/                    # 40,000 images (80% training split)
│   ├── val/                      # 10,000 images (20% validation split)
│   └── test/                     # 10,000 images (unbiased evaluation set)
├── docs/
│   └── architecture_decision_log.md  # Architectural decisions & design rationale
├── models/                       # Ignored by git
│   └── baseline_resnet18_best.pt # Saved weights (Best Val Acc: 80.56%)
├── results/
│   └── training_metrics.csv      # Epoch-by-epoch loss & accuracy logs
├── src/
│   ├── __init__.py
│   ├── data_pipeline.py          # Data download, split, and export script
│   └── train.py                  # ResNet-18 baseline training & validation script
├── .gitignore                    # ML-oriented ignore rules
├── requirements.txt              # Pinned Python dependencies
└── README.md                     # Project documentation
```

---

## Setup & Quickstart

### 1. Prerequisites
- Python 3.10+ (Tested on Python 3.14)
- Git

### 2. Environment Setup

Clone the repository and initialize a virtual environment:

```bash
# Clone repository
git clone <repo-url>
cd bright-future-image-classifier

# Create virtual environment
python -m venv bright_future

# Activate virtual environment
# On Windows (PowerShell):
.\bright_future\Scripts\Activate.ps1
# On Linux/macOS:
source bright_future/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### 3. Run Data Ingestion Pipeline

To download the baseline CIFAR-10 dataset and generate the reproducible train/val/test splits:

```bash
python src/data_pipeline.py
```

### 4. Train the Baseline Model (Task 2)

To train the pre-trained ResNet-18 baseline model for 10 epochs using Adam optimizer:

```bash
python src/train.py --epochs 10 --batch-size 128 --lr 0.001
```

---

## Task 2 Baseline Evaluation Results

Training was conducted for 10 epochs on the 40,000 training images and evaluated on 10,000 validation images. Logs are stored in `results/training_metrics.csv`.

| Epoch | Train Loss | Train Accuracy | Validation Loss | Validation Accuracy | Status |
| :---: | :---: | :---: | :---: | :---: | :---: |
| **1** | 0.9399 | 68.11% | 0.7719 | 74.65% | Target Exceeded ($\ge 70\%$) |
| **2** | 0.5961 | 79.88% | 0.7152 | 76.10% | |
| **3** | 0.4671 | 84.19% | 0.6116 | 79.79% | |
| **4** | 0.3667 | 87.59% | 0.6551 | 78.58% | |
| **5** | 0.2888 | 90.26% | 0.6201 | 80.47% | |
| **6** | 0.2219 | 92.57% | 0.6772 | 79.89% | |
| **7** | 0.1868 | 93.65% | 0.6883 | 80.43% | |
| **8** | 0.1554 | 94.72% | 0.7062 | 80.48% | |
| **9** | 0.1261 | 95.73% | 0.7394 | 79.90% | |
| **10** | 0.1019 | 96.50% | 0.7616 | **80.56%** | **★ Best Val Accuracy** |

> **Key Takeaway:** The pre-trained ResNet-18 achieved **80.56% validation accuracy**, surpassing the client target of 70% by over **10.5%** in the baseline configuration.

---

## Project Roadmap

- [x] **Task 1:** Repository setup, data pipeline, and architecture decision log.
- [x] **Task 2:** Build and evaluate transfer-learning baseline model (ResNet-18).
- [ ] **Task 3:** Data augmentation, class weighting, and hyperparameter optimization.
- [ ] **Task 4:** Error analysis and Grad-CAM visual explanations.
- [ ] **Task 5:** Containerize FastAPI inference service and deploy to Render.
- [ ] **Task 6:** Final handover package, model card, and user guide.
