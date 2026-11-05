# Bright Future School: Classroom vs Playground Image Classifier

Automated computer vision classification system developed for **Bright Future School**. The system automatically categorizes event photos into **classroom** and **playground** scenes, streamlining digital media organization for school newsletters, archives, and social media.

---

## Project Overview
- **Client:** Bright Future School
- **Objective:** Build, evaluate, and deploy an image classification pipeline achieving at least **70% accuracy**.
- **Current Phase:** Task 1 – Project Repository Setup, Data Pipeline, and Architecture Baseline.

---

## Task 1 Deliverables
- Modular, production-ready directory layout (`src/`, `data/`, `docs/`).
- Automated data ingestion and 3-way split pipeline script (`train/`, `val/`, `test/`).
- Architecture Decision Log (ADL) documenting model selection, split rationale, and preprocessing strategy.
- Git configuration with comprehensive `.gitignore` for ML artifacts and datasets.

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
├── src/
│   ├── __init__.py
│   └── data_pipeline.py          # Data download, split, and export script
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

### 3. Run the Data Pipeline

To download the baseline CIFAR-10 dataset, partition it into deterministic train/val/test splits, and export images into an `ImageFolder`-compatible structure:

```bash
python src/data_pipeline.py
```

Optional CLI flags:
```bash
# Re-run and force re-export even if data exists
python src/data_pipeline.py --force

# Custom split ratio (e.g. 85% train, 15% val)
python src/data_pipeline.py --val-ratio 0.15 --seed 42
```

### 4. Dataset Partition Summary

| Split | Images | Ratio | Purpose |
| :--- | :---: | :---: | :--- |
| **Train** | 40,000 | 80% of Train partition | Model parameter learning |
| **Val** | 10,000 | 20% of Train partition | Hyperparameter tuning & model selection |
| **Test** | 10,000 | Dedicated test split | Unbiased final performance benchmark |

---

## Project Roadmap

- [x] **Task 1:** Repository setup, data pipeline, and architecture decision log.
- [ ] **Task 2:** Build and evaluate transfer-learning baseline model (ResNet-18).
- [ ] **Task 3:** Data augmentation, class weighting, and hyperparameter optimization.
- [ ] **Task 4:** Error analysis and Grad-CAM visual explanations.
- [ ] **Task 5:** Containerize FastAPI inference service and deploy to Render.
- [ ] **Task 6:** Final handover package, model card, and user guide.
