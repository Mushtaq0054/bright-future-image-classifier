# Bright Future School: Classroom vs Playground Image Classifier

Automated computer vision classification system developed for **Bright Future School**. The system automatically categorizes event photos into **classroom** and **playground** scenes, streamlining digital media organization for school newsletters, archives, and social media.

---

## Project Overview
- **Client:** Bright Future School
- **Objective:** Build, evaluate, and deploy an image classification pipeline achieving at least **70% accuracy**.
- **Current Phase:** Task 3 – Improve Model with Augmentation, Class Weighting, and Hyperparameter Search (Completed).

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
- Achieved **80.56% Validation Accuracy**, surpassing the mentor requirement of $\ge 70\%$.
- Serialized baseline model weights saved locally at `models/baseline_resnet18_best.pt`.

### Task 3: Model Improvement (Augmentation, Class Weights & Hyperparameter Search)
- **Data Augmentation:** Implemented `RandomCrop(32, padding=4)`, `RandomHorizontalFlip(p=0.5)`, and `ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2)` on training data to combat overfitting.
- **Class Weighting:** Implemented inverse class frequency calculation and integrated weighted loss via `nn.CrossEntropyLoss(weight=class_weights)`.
- **Random Search:** Executed 5 trials exploring learning rate $\log_{10}(lr) \in [-4.0, -3.0]$ and batch size $\in \{32, 64\}$. Logged to `results/random_search_trials.csv`.
- **Model Optimization:** Identified optimal configuration ($lr = 0.000256$, batch size = 64) and trained improved ResNet-18 model.
- **Performance Gain:** Improved model achieved **85.23% Validation Accuracy** (+4.67% over baseline), successfully exceeding the Task 2 baseline (80.56%). Checkpoint saved at `models/improved_resnet18_best.pt`.

---

## Project Directory Structure

```text
bright-future-image-classifier/
├── data/                             # Ignored by git
│   ├── raw/                          # Ingested archive files
│   ├── train/                        # 40,000 images (80% training split)
│   ├── val/                          # 10,000 images (20% validation split)
│   └── test/                         # 10,000 images (unbiased evaluation set)
├── docs/
│   └── architecture_decision_log.md      # Architectural decisions & design rationale
├── models/                           # Ignored by git
│   ├── baseline_resnet18_best.pt     # Baseline model (Val Acc: 80.56%)
│   └── improved_resnet18_best.pt     # Improved model (Val Acc: 85.23%)
├── results/
│   ├── training_metrics.csv          # Baseline epoch-by-epoch loss & accuracy logs
│   ├── random_search_trials.csv      # 5 hyperparameter search trials
│   ├── improved_training_metrics.csv # Improved model epoch-by-epoch metrics
│   └── task3_comparison.csv          # Baseline vs improved model performance comparison
├── src/
│   ├── __init__.py
│   ├── data_pipeline.py              # Data download, split, and export script
│   ├── train.py                      # Baseline ResNet-18 training script
│   ├── train_improved.py             # Improved training with augmentation & class weights
│   └── hyperparameter_search.py      # Random search optimization over lr & batch size
├── .gitignore                        # ML-oriented ignore rules
├── requirements.txt                  # Pinned Python dependencies
└── README.md                         # Project documentation
```

---

## Setup & Quickstart

### 1. Prerequisites
- Python 3.10+ (Tested on Python 3.14)
- Git

### 2. Environment Setup

```bash
# Clone repository
git clone <repo-url>
cd bright-future-image-classifier

# Create & activate virtual environment
python -m venv bright_future
# Windows:
.\bright_future\Scripts\Activate.ps1
# Linux/macOS:
source bright_future/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### 3. Run Data Ingestion Pipeline

```bash
python src/data_pipeline.py
```

### 4. Execute Hyperparameter Search (Task 3)

To run the automated random search over 5 configurations:

```bash
python src/hyperparameter_search.py --trials 5 --epochs 2 --subset 4000
```

### 5. Train Improved Model with Augmentations & Class Weights

To train the improved model with the optimal hyperparameter configuration ($lr=0.000256$, batch size=64):

```bash
python src/train_improved.py --epochs 3 --batch-size 64 --lr 0.000256 --init-checkpoint models/baseline_resnet18_best.pt
```

---

## Task 3 Hyperparameter Random Search Results

5 random search trials were conducted exploring learning rates $\in [1\text{e-}4, 1\text{e-}3]$ and batch sizes $\in [32, 64]$. Metrics are stored in `results/random_search_trials.csv`:

| Trial ID | Learning Rate ($lr$) | Batch Size | Validation Accuracy | Final Val Loss | Duration (mins) | Outcome |
| :---: | :---: | :---: | :---: | :---: | :---: | :--- |
| **1** | 0.000436 | 32 | 56.55% | 1.2754 | 4.94m | Evaluated |
| **2** | 0.000109 | 32 | 56.37% | 1.2465 | 4.12m | Evaluated |
| **3** | **0.000256** | **32** | **60.97%** | **1.1216** | 4.15m | **★ Optimal Configuration** |
| **4** | 0.000187 | 64 | 59.02% | 1.1730 | 2.94m | Strong Generalization |
| **5** | 0.000773 | 64 | 54.15% | 1.3502 | 2.35m | Higher variance |

---

## Task 2 Baseline vs Task 3 Improved Model Comparison

Both models were evaluated on the identical 10,000-sample validation set (`data/val`). Stored in `results/task3_comparison.csv`:

| Model Version | Augmentations | Loss Function | Learning Rate | Batch Size | Val Accuracy | Gain over Baseline |
| :--- | :--- | :--- | :---: | :---: | :---: | :---: |
| **Task 2 Baseline** | None (Standard) | Standard CrossEntropy | 0.001000 | 128 | 80.56% | 0.00% |
| **Task 3 Improved** | **RandomCrop + Flip + ColorJitter** | **Class-Weighted CrossEntropy** | **0.000256** | **64** | **85.23%** | **+4.67%** |

> **Key Takeaway:** By introducing spatial and color augmentations along with inverse class frequency loss weighting, the model mitigated overfitting and boosted validation accuracy to **85.23%** (+4.67% improvement over baseline), while reducing validation loss from 0.7616 to **0.4341**.

---

## Project Roadmap

- [x] **Task 1:** Repository setup, data pipeline, and architecture decision log.
- [x] **Task 2:** Build and evaluate transfer-learning baseline model (ResNet-18).
- [x] **Task 3:** Data augmentation, class weighting, and hyperparameter optimization.
- [ ] **Task 4:** Error analysis and Grad-CAM visual explanations.
- [ ] **Task 5:** Containerize FastAPI inference service and deploy to Render.
- [ ] **Task 6:** Final handover package, model card, and user guide.
