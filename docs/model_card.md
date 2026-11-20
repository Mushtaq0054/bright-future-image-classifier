# Model Card: ResNet-18 Image Classifier for Bright Future School

Following the standard Model Card framework (*Mitchell et al., 2019*).

---

## 1. Model Details

- **Model Name:** Bright Future Image Classifier (ResNet-18 v2)
- **Model Version:** 2.0.0 (Fine-Tuned with Augmentations & Class Weighting)
- **Model Type:** Deep Convolutional Neural Network (CNN)
- **Base Architecture:** ResNet-18 (He et al., 2016) with residual skip connections
- **Pretrained Weights:** ImageNet-1k (Torchvision `ResNet18_Weights.DEFAULT`)
- **Adapted Head:** Linear classifier (`in_features=512, out_features=10`)
- **Developers:** Mushtaq Ahmad (Machine Learning Intern, Bright Future School project)
- **Release Date:** November 2026
- **License:** MIT License
- **Framework:** PyTorch 2.15.0 / Torchvision 0.29.2
- **Model Checkpoint Path:** `models/improved_resnet18_best.pt` (File size: ~42.7 MB)

---

## 2. Intended Use

### Primary Intended Uses
- **Operational Goal:** Automated tagging and organization of school event photographs into **classroom** (indoor learning, laboratory, activities) and **playground** (outdoor sports, fields, assemblies) categories.
- **Client Application:** Integration with Bright Future School's digital asset management portal, automated newsletter curation tools, and social media media library.
- **Inference Mode:** Real-time single image upload and batch processing via FastAPI REST API (`/predict`).

### Out-of-Scope & Unintended Uses
- **Biometric Identification:** Not intended, trained, or licensed for facial recognition, student identity tracking, or surveillance.
- **Safety-Critical Automation:** Should not be used for security monitoring, perimeter breach detection, or medical diagnostics without human oversight.
- **High-Resolution Medical/Forensic Imaging:** Trained on standard natural resolution imagery ($32 \times 32$ to $224 \times 224$ downsampled).

---

## 3. Factors & Context

- **Lighting Conditions:** Trained on varied color-jittered images (brightness, contrast, saturation) to handle both bright outdoor sunlight and indoor fluorescent classroom lighting.
- **Image Resolution:** Optimized for low-to-medium resolution imagery downscaled to $32 \times 32$ spatial tensor input with standard ImageNet channel normalization:
  $$\mu = [0.485, 0.456, 0.406], \quad \sigma = [0.229, 0.224, 0.225]$$
- **Scene Composition:** Expects centered, photographic event imagery. High-clutter images with occlusions may experience degraded classification confidence.

---

## 4. Performance Metrics

### Evaluation Results (Validation & Test Sets)
| Metric | Task 2 Baseline | Task 3 Improved Model | Target Threshold | Status |
|:---|:---:|:---:|:---:|:---:|
| **Validation Accuracy** | 80.56% | **85.23%** | $\ge 70.00\%$ | **Passed (+15.23% above target)** |
| **Test Accuracy (Unseen)** | — | **84.66%** | $\ge 70.00\%$ | **Passed (+14.66% above target)** |
| **Validation Loss** | 0.7616 | **0.4341** | — | **Reduced by 43.0%** |
| **Inference Latency (CPU)** | ~18 ms | **~14 ms** | $< 100\text{ ms}$ | **Passed** |
| **Model Size** | 42.7 MB | **42.7 MB** | $< 100\text{ MB}$ | **Free-tier Compliant** |

### Per-Class Test Accuracy Breakdown (10,000 Unseen Samples):
- `automobile`: 92.40%
- `ship`: 90.90%
- `frog`: 90.20%
- `horse`: 88.70%
- `airplane`: 87.10%
- `truck`: 84.00%
- `deer`: 83.90%
- `bird`: 79.40%
- `dog`: 78.20%
- `cat`: 71.80%

---

## 5. Training & Evaluation Data

- **Benchmark Proxy Dataset:** Programmatic 3-way partitioned CIFAR-10 proxy benchmark (ADR-001):
  - **Training Set:** 40,000 images (80% split)
  - **Validation Set:** 10,000 images (20% split)
  - **Test Set:** 10,000 images (unbiased external test partition)
- **Data Augmentation:**
  - `RandomCrop(32, padding=4)`
  - `RandomHorizontalFlip(p=0.5)`
  - `ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2)`
- **Loss Function:** Class-weighted Cross-Entropy loss computed via inverse class frequency to penalize difficult categories:
  $$w_c = \frac{N}{C \times N_c}$$

---

## 6. Quantitative Analysis & Error Patterns

As documented in [`docs/error_analysis_report.md`](error_analysis_report.md) and [`results/misclassified_samples.csv`](../results/misclassified_samples.csv):
1. **Background Correlation Bias:** In low-resolution imagery, green foliage/grass textures occasionally cause domestic animals (`cat`, `dog`) to be misclassified as `frog` (e.g. `img_05511.png` at 99.98% confidence).
2. **Pose & Silhouette Overlap:** Quadruped animals viewed in upright profile mimic equine limb geometry, causing occasional `dog`-as-`horse` misclassifications.
3. **Vehicle Boundary Ambiguity:** Light-duty pickup trucks share frontal hood and cabin geometry with standard passenger automobiles.

---

## 7. Explainability & Interpretability (Grad-CAM)

- **Method:** Gradient-weighted Class Activation Mapping (Grad-CAM) applied to `model.layer4` (the final 512-channel convolutional block).
- **Mathematical Formula:**
  $$\alpha_k^c = \frac{1}{Z} \sum_{i,j} \frac{\partial y^c}{\partial A_{i,j}^k}, \quad L_{\text{Grad-CAM}}^c = \text{ReLU}\left( \sum_k \alpha_k^c A^k \right)$$
- **Artifacts:** 3-panel diagnostic figures `[Original | Heatmap | Overlay]` available in `results/gradcam/`.

---

## 8. Ethical Considerations & Operational Recommendations

1. **Human-in-the-Loop Safeguard:** Images with model confidence $< 70\%$ or top-1/top-2 entropy margin $< 0.15$ should be routed to an `Uncertain / Review` folder for staff confirmation.
2. **Privacy by Design:** The API does not persist or log uploaded image bytes to permanent storage; images are decoded in-memory (`BytesIO`) and discarded immediately post-inference.
3. **Continuous Retraining:** Recommended quarterly fine-tuning with school-specific photos using transfer learning to align background feature distributions with the actual school campus.
