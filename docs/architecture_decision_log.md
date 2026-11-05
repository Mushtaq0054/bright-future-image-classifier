# Architecture Decision Log (ADL)

## Project: Bright Future School Image Classification
**Task 1: Repository Setup, Data Pipeline, and Architecture Baseline**

### 1. Context & Business Problem
Bright Future School ko school events ki media library organize karne ke liye automated image classifier chahiye (classroom vs playground scenes target scenario).

### 2. Dataset Selection & Ingestion
* **Dataset Choice:** Torchvision standard CIFAR-10 dataset.
* **Rationale:** Automated, programmatic download ke zariye reproducible baseline benchmark tayyar karna jo bina manual interventions ke cross-platform verify ho sake.
* **Storage Pattern:** Local `data/raw/` directory for archives and structured class directories (`airplane/`, `dog/`, etc.) in target splits.

### 3. Data Splitting Strategy
* **Total Training Set (50,000 images):**
  * **Train Set:** 40,000 images (80%) model parameter learning ke liye.
  * **Validation Set:** 10,000 images (20%) tuning aur early evaluation ke liye.
* **Test Set:** 10,000 images (standard torchvision test partition) final unbiased performance check ke liye.
* **Reproducibility:** PyTorch `Generator().manual_seed(42)` fix kiya gaya hai taaki splits hamesha identical rahen.

### 4. Target Model Architecture (Task 2 & 3 Road Map)
* **Model Choice:** Pretrained ResNet-18 (ImageNet weights).
* **Rationale:** Deep residual connections vanishing gradient problem ko solve karti hain aur small-to-medium datasets par scratch training se behtar feature representations deti hain.
* **Final Layer:** Task 2 me classification head ko 10 classes ke mutabiq adapt kiya jayega.

### 5. Input Preprocessing & Augmentation Strategy
* **Task 1 Baseline:** PyTorch `ImageFolder` compatible folder hierarchy create ki gayi hai.
* **Planned Augmentation (Task 3):** Random Crop, Horizontal Flip, aur Color Jitter overfitting rokne ke liye design kiye jayenge.