# Bright Future School: Classroom vs Playground Image Classifier

Automated computer vision classification system developed for **Bright Future School**. The system automatically categorizes event photos into **classroom** and **playground** scenes, streamlining digital media organization for school newsletters, archives, and social media.

---

## Project Overview
- **Client:** Bright Future School
- **Objective:** Build, evaluate, and deploy an image classification pipeline achieving at least **70% accuracy**.
- **Current Phase:** Task 4 – Perform Error Analysis and Add Grad-CAM Visual Explanations (Completed).

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

### Task 4: Error Analysis & Grad-CAM Visual Explainability
- **Automated Test Error Audit:** Evaluated improved ResNet-18 model on 10,000 unseen test images (`src/error_analysis.py`). Achieved **84.66% Overall Test Accuracy** (8,466 correct vs. 1,534 misclassified).
- **Misclassified Samples Tracking:** Generated comprehensive CSV export [`results/misclassified_samples.csv`](results/misclassified_samples.csv) detailing image paths, true vs predicted labels, and top-2 prediction confidences.
- **Grad-CAM Implementation:** Developed PyTorch-native hook architecture (`src/gradcam.py`) extracting gradients and activations from `model.layer4` to compute class-discriminative saliency heatmaps.
- **Visual Diagnostic Figures:** Generated 3-panel PNG diagnostic visualizations `[Original Image | Grad-CAM Heatmap | Overlay]` in [`results/gradcam/`](results/gradcam/).
- **Error Analysis Report:** Authored in-depth technical analysis [`docs/error_analysis_report.md`](docs/error_analysis_report.md) breaking down root causes (spurious background correlations, quadruped limb ambiguity, vehicle silhouette overlap) and actionable safety thresholds for Task 5 deployment.

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
│   ├── architecture_decision_log.md      # Architectural decisions & design rationale
│   └── error_analysis_report.md          # Comprehensive error analysis & Grad-CAM findings
├── models/                           # Ignored by git
│   ├── baseline_resnet18_best.pt     # Baseline model (Val Acc: 80.56%)
│   └── improved_resnet18_best.pt     # Improved model (Val Acc: 85.23%, Test Acc: 84.66%)
├── results/
│   ├── training_metrics.csv          # Baseline epoch-by-epoch loss & accuracy logs
│   ├── random_search_trials.csv      # 5 hyperparameter search trials
│   ├── improved_training_metrics.csv # Improved model epoch-by-epoch metrics
│   ├── task3_comparison.csv          # Baseline vs improved model performance comparison
│   ├── misclassified_samples.csv     # Complete ranked list of 1,534 test misclassifications
│   └── gradcam/                      # Grad-CAM 3-panel diagnostic figures
│       ├── gradcam_error_01_truck_as_automobile.png
│       ├── gradcam_error_02_dog_as_horse.png
│       ├── gradcam_error_03_cat_as_frog.png
│       ├── gradcam_error_04_truck_as_automobile.png
│       └── gradcam_error_05_cat_as_frog.png
├── src/
│   ├── __init__.py
│   ├── data_pipeline.py              # Data download, split, and export script
│   ├── train.py                      # Baseline ResNet-18 training script
│   ├── train_improved.py             # Improved training with augmentation & class weights
│   ├── hyperparameter_search.py      # Random search optimization over lr & batch size
│   ├── error_analysis.py             # Test set evaluation and misclassification extraction
│   └── gradcam.py                    # PyTorch Grad-CAM computation & visualization
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

### 6. Perform Error Analysis (Task 4)

To evaluate the improved model on the 10,000-image test set and generate the misclassification CSV:

```bash
python src/error_analysis.py --model-path models/improved_resnet18_best.pt --split test
```

### 7. Generate Grad-CAM Heatmaps (Task 4)

To generate 3-panel Grad-CAM diagnostic figures for high-confidence failure cases:

```bash
python src/gradcam.py --count 5
```

---

## Task 4 Error Analysis & Grad-CAM Findings

The improved ResNet-18 model was thoroughly evaluated on the independent 10,000-image test set:
- **Test Accuracy:** **84.66%** (8,466 correct / 1,534 misclassified).
- **Misclassified Log:** All 1,534 failure cases are catalogued and sorted by prediction confidence in [`results/misclassified_samples.csv`](results/misclassified_samples.csv).
- **Comprehensive Report:** See [`docs/error_analysis_report.md`](docs/error_analysis_report.md) for full case studies and deployment safety recommendations.

### Key Grad-CAM Failure Insights:
1. **Background / Context Shortcuts:** Cats on wet green lawns strongly trigger `frog` predictions because the model attends to moss/grass texture rather than feline facial anatomy.
2. **Structural & Silhouette Overlap:** Dogs standing upright in side profile activate equine (`horse`) features due to limb length and stance geometry.
3. **Sub-category Confusion:** Trucks viewed from three-quarter angles without a distinct visible cargo bed activate automobile cabin and chassis filters.

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
- [x] **Task 4:** Error analysis and Grad-CAM visual explanations.
- [ ] **Task 5:** Containerize FastAPI inference service and deploy to Render.
- [ ] **Task 6:** Final handover package, model card, and user guide.
