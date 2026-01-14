# 📘 Iso-Norm Sparse SAM: Correcting Norm Collapse

**A Robust Regularization Framework for High-Sparsity Optimization**

This repository provides the official implementation of **Iso-Norm Sparse SAM (SSAM)**. It identifies and solves the "norm collapse" problem inherent in standard sparse Sharpness-Aware Minimization (SAM), where high-sparsity masks inadvertently destroy the regularization effect.

The framework integrates:

* **Standard Sparse SAM (SSAM-F)** as a baseline.
* **Iso-Norm Correction** to maintain perturbation strength at high sparsity.
* **ResNet-18 (CIFAR-optimized)** architecture.
* **Stratified Subsetting** to simulate data scarcity.

The goal is to enable efficient, sparse regularization without sacrificing generalization performance.

---

## 🧱 Architecture Overview

### Model Level

* **Backbone:** ResNet-18 (Modified for CIFAR-10)
* *Modification:* Removed initial 7x7 max-pooling to preserve spatial resolution for 32x32 images.


* **Dataset:** CIFAR-10 (50% Stratified Subset)
* *Purpose:* Simulates data-scarce environments where regularization is most critical.



### Optimization Level

* **Base Optimizer:** SGD with Momentum (0.9)
* **Meta Optimizer:** Sparse SAM (Sharpness-Aware Minimization)
* **Ascent Step:** Sparse gradient perturbation (Top-k filtering).
* **Correction Step:** **Iso-Norm Scaling** (Proposed).
* **Descent Step:** Standard weight update.



### The Core Problem & Solution

1. **Standard SSAM (The Problem):** When 95% of the perturbation vector is masked to zero, the Euclidean norm () drops significantly. The actual perturbation radius becomes much smaller than the target , leading to weak regularization ("Norm Collapse").
2. **Iso-Norm SSAM (The Solution):** We explicitly re-scale the sparse perturbation vector to force its length to equal the target , regardless of sparsity level.

---

## 🚀 Regularization Techniques

### 1. Sharpness-Aware Minimization (SAM)

Minimizes both loss value and loss sharpness, encouraging the model to find flat minima which generalize better.

### 2. Sparse Perturbation (Top-k)

Instead of perturbing every weight in the network, we only perturb the top  weights with the largest gradients. This reduces computational overhead and focuses regularization on the most sensitive parameters.

### 3. Iso-Norm Correction (Proposed)

A mathematically grounded rescaling step that prevents the effective perturbation radius from shrinking as sparsity increases.


---

## 🔧 Training Pipeline Overview

**Per Training Iteration:**

1. **Forward Pass:** Compute standard loss.
2. **Backward Pass:** Compute gradients.
3. **Ascent Step (Sparse SAM):**
* Calculate dense perturbation.
* Apply Top-k mask (95% sparsity).
* **Apply Iso-Norm Correction (if enabled).**
* Update weights with perturbation.


4. **Second Forward/Backward:** Compute gradient at perturbed state.
5. **Descent Step:** Update original weights using perturbed gradient.

**Per Epoch:**

1. Compute Train/Test Accuracy and Loss.
2. **Log Perturbation Norm:** Verify if  matches target .
3. Calculate **Generalization Gap** (Train Acc - Test Acc).

---

## ⚙️ Hyperparameters Summary

| Component | Value | Notes |
| --- | --- | --- |
| **Model** | ResNet-18 | CIFAR-10 variant |
| **Dataset Size** | 50% | Stratified subset |
| **Batch Size** | 128 |  |
| **Epochs** | 100 | Cosine Annealing |
| **Learning Rate** | 0.05 |  |
| **Rho ()** | 0.05 | Target neighborhood size |
| **Sparsity** | 95% | 0.95 (Top-5% active) |
| **Seeds** | `[8, 42, 123]` | For statistical validity |

---

## 🧪 Experiments (Included in Code)

To validate the method, the provided script runs these four comparative experiments:

| ID | Experiment | Optimizer | Sparsity | Norm Correction | Status |
| --- | --- | --- | --- | --- | --- |
| **E1** | **SGD** | SGD | N/A | N/A | Baseline |
| **E2** | **Dense SAM** | SAM | 0% | No | Gold Standard |
| **E3** | **Standard SSAM** | Sparse SAM | 95% | **No** | **Fails (Collapses)** |
| **E4** | **Iso-Norm SSAM** | Sparse SAM | 95% | **Yes** | **Succeeds** |

**E4 is the proposed method.**

---

## 📊 Plots Required for Research Paper

The repository includes an automated plotting suite that generates:

1. **Test Accuracy Curve (Mean ± Std)**
* *Purpose:* Shows Iso-Norm recovering the performance lost by Standard SSAM.


2. **Perturbation Norm Verification**
* *Purpose:* The "Smoking Gun" proof. Shows Standard SSAM norms dropping to , while Iso-Norm stays perfectly at .


3. **Generalization Gap Plot**
* *Purpose:* Demonstrates that Iso-Norm minimizes the gap between Train and Test accuracy better than the baseline.


4. **Final Accuracy Bar Chart**
* *Purpose:* A clean summary comparison for the "Results" section.


5. **Loss Curves (Train/Test)**
* *Purpose:* Standard convergence verification.



---

## 📈 Metrics You Must Report

### Core Metrics

* **Test Accuracy:** Final accuracy on the hold-out set.
* **Perturbation Norm:** The actual measured L2 norm of the perturbation vector during training.
* **Generalization Gap:** (Train Acc - Test Acc). Lower is better.

### Optional Metrics

* **Training Time:** Seconds per epoch (to show overhead vs dense SAM).
* **Loss Sharpness:** Hessian trace (if available).

---

## 📁 Recommended Folder Structure

```text
iso_norm_ssam/
├── data/                   # CIFAR-10 downloads here
├── sparse_sam_results/     # JSON logs for every seed
│   ├── sgd_s8_sp95.json
│   ├── ssam_isonorm_s8_sp95.json
│   └── ...
├── figures_comparison/     # Generated Plots
│   ├── 1_test_acc.png
│   ├── 2_gen_gap.png
│   └── 3_perturb_norms.png
├── train.py                # Main training script
├── plot.py                 # Plotting suite
└── README.md

```

---

## 🎉 Summary

This repository provides a mathematically grounded fix for sparse SAM algorithms. By enforcing **Iso-Norm constraints**, we allow models to benefit from the efficiency of high sparsity (95%) without suffering from the performance degradation caused by perturbation shrinking.

The method is **simple, statistically robust (validated across 3 seeds), and effective.**

---
