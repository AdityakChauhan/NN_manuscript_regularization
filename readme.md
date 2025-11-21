# 📘 Regularization Framework for CNN Training + EMA-Smoothed CSDC Early Stopping

A Unified Overfitting Prevention Pipeline for CNNs

This repository provides a **complete, unified regularization framework** for training CNNs on image datasets (CIFAR-10, CIFAR-100, Caltech-101, TinyImageNet). The pipeline integrates:

* **Strong augmentations (Mixup / CutMix)**
* **Stochastic Depth (Drop Path)**
* **Soft Target Cross Entropy**
* **SAM + AdamW optimizer**
* **EMA-Smoothed CSDC Early Stopping**

The goal is to prevent overfitting while keeping the method **simple, reproducible, and effective**.

---

# 🧱 Architecture Overview

### **Data Level**

* Standard Augmentations:

  * `RandomResizedCrop`
  * `RandomHorizontalFlip`
  * `Normalize`
* Mixup (via timm)
* CutMix (via timm)

### **Model Level**

* ResNet / ConvNeXt / similar
* Stochastic Depth (Drop Path)

### **Optimization Level**

* SAM optimizer (two-step optimization)
* Inner optimizer: AdamW

### **Loss Level**

* Soft Target Cross Entropy (supports mixed soft labels)

### **Early Stopping**

* EMA-Smoothed CSDC (Correlation-Driven Stopping Criterion)

---

# 🚀 Regularization Techniques

### **1. Mixup (timm)**

Blends inputs & labels → reduces memorization.

### **2. CutMix (timm)**

Patches images → improves generalization.

### **3. Soft Target Cross Entropy**

Smooth, stable gradients & compatible with augmented labels.

### **4. Stochastic Depth**

Drop full residual paths randomly → prevents co-adaptation.

### **5. SAM + AdamW**

* SAM: encourages flat minima → better generalization
* AdamW: stable weight decay

### **6. Standard Augmentations**

Enhance robustness & reduce sensitivity to input variance.

---

# ⏹ EMA-CSDC: Early Stopping Reformulated

### **Steps:**

1. After each epoch store training & validation loss.
2. Smooth both with EMA.
3. Compute Pearson correlation over the last `ω` epochs.
4. If correlation < μ (threshold): divergence detected.
5. If this triggers λ times: stop training.
6. Restore best validation model.

### **Recommended settings**

* Window (ω): 10–20
* Threshold (μ): 0.0
* Patience (λ): 3–5

---

# 🔧 Training Pipeline Overview

### **Per training iteration**

1. Load batch
2. Apply augmentations
3. Apply Mixup / CutMix
4. Forward pass
5. Compute Soft Target CE loss
6. SAM First Step
7. SAM Second Step
8. AdamW update

### **Per epoch**

* Compute train & val loss
* Smooth using EMA
* Run CSDC correlation
* Save best model
* Stop if conditions met

---

# ⚙️ Hyperparameters Summary

| Component          | Value            |
| ------------------ | ---------------- |
| Mixup α            | 0.2              |
| CutMix probability | 0.5              |
| Drop Path          | 0.1–0.2          |
| Optimizer          | SAM + AdamW      |
| LR                 | 0.001            |
| Weight Decay       | 1e-4             |
| Scheduler          | Cosine Annealing |
| Loss               | Soft Target CE   |
| CSDC Window        | 10–20            |
| CSDC Threshold     | 0.0              |
| CSDC Patience      | 3–5              |

---

# 📦 Supported Datasets

* CIFAR-10
* CIFAR-100
* Caltech-101
* TinyImageNet

Use **CIFAR-10** for clean demonstration.

---

# 🧪 Experiments You MUST Train (for the Paper)

To show your method works, train these five experiments:

| ID | Experiment            | Model     | Regularization        | Stopping       |
| -- | --------------------- | --------- | --------------------- | -------------- |
| E1 | Baseline              | ResNet-18 | None                  | None           |
| E2 | Mixup + CutMix        | ResNet-18 | Mixup+CutMix          | None           |
| E3 | + Stochastic Depth    | ResNet-18 | Mixup+CutMix+DropPath | None           |
| E4 | Full Reg + Early Stop | ResNet-18 | Full Reg              | Early Stopping |
| E5 | **Your Method**       | ResNet-18 | Full Reg              | **EMA-CSDC**   |

E5 is your final proposed system.

---

# 📊 Plots Required for Research Paper (Must Include)

### **1. Accuracy Curve (Train vs Test)**

Shows overfitting reduction.

### **2. Loss Curve (Train vs Validation)**

Shows stability under CSDC.

### **3. CSDC Correlation Curve**

Correlation vs Epoch → highlight the stopping point.

### **4. Bar Chart: Accuracy Across E1–E5**

Shows impact of each regularization layer.

### **5. Ablation Table**

Which component contributes how much.

### **6. Overfitting Gap Plot**

Gap = Train Acc − Test Acc.

### **7. Confusion Matrix**

Final classification quality.

### **8. Sharpness Visualization (Optional)**

SAM vs non-SAM minima.

### **9. Soft Labels Histogram (Mixup/CutMix)**

Visualize soft target distribution.

---

# 📈 Metrics You Must Report

### **Core Metrics**

* Train Accuracy
* Test Accuracy
* Train Loss
* Validation Loss
* Overfitting Gap
* Epoch of CSDC stop

### **Optional but Strong Metrics**

* Mean Correlation Value
* Label Entropy (shows soft label behavior)
* Sharpness Measure (trace of Hessian approximation)
* Inference Robustness (noise tests)

---

# 📁 Recommended Folder Structure

```
project_root/
├── models/
├── trainers/
├── datasets/
├── utils/
├── results/
│   ├── logs/
│   ├── curves/
│   ├── correlations/
│   ├── confusion_matrices/
│   └── checkpoints/
└── README.md
```

---

# 🎉 Summary

This repository provides a **complete overfitting prevention system** built on:

* Mixup + CutMix
* Stochastic Depth
* Soft Target CE
* SAM + AdamW optimization
* EMA-CSDC stopping

The method is simple, robust, and generalizes strongly across datasets.

---

# 📧 Contact

Open an issue for questions, reproduction difficulties, or feature requests.
