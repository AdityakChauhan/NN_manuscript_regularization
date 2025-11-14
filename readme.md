# 📘 Regularization Framework for CNN Training + CSDC Early Stopping

This repository implements a **practical and effective regularization framework** for training CNNs on image datasets such as **CIFAR-10**, **TinyImageNet**, and **Caltech-101**, paired with a modern stopping method called **CSDC (Correlation-Driven Stopping Criterion)**.

The goal is to prevent overfitting and improve generalization while keeping the method **clear, reproducible, and simple**.

---

# 🚀 Regularization Techniques Used During Training

### **1. Mixup**
Prevents memorization by blending two images and their labels.

### **2. CutMix**
Improves robustness by cutting and pasting image patches.

### **3. Label Smoothing**
Reduces model overconfidence and stabilizes training.

### **4. Adaptive Dropout**
Adds controlled noise inside the network to prevent co-adaptation.
Dropout probability: `0.2`.

### **5. SAM + AdamW**
- SAM encourages flat minima → better generalization.
- AdamW provides stable weight decay.

### **6. Standard Augmentations**
- RandomResizedCrop  
- RandomHorizontalFlip  
- Normalize  

These operate at the **input level**.

---

# ⏹️ Early Stopping Method: CSDC (Correlation-Driven Stopping Criterion)

CSDC is a stopping method designed to detect when **training loss and validation loss begin to diverge**, which is an early sign of overfitting.

### **How CSDC Works**
After every epoch:

1. Store training loss `e_tr` and validation loss `e_va`.
2. Compute the **rolling Pearson correlation** over a window of length `ω`:r = corr( e_tr[n−ω : n], e_va[n−ω : n] )


3. If the correlation `r` falls below a threshold `μ`,  
→ this means losses are diverging.
4. If this happens for `λ` total epochs (patience),  
→ **training stops**.
5. The best validation model so far is selected.

### **Hyperparameters**
- Window size (ω): 10–20  
- Correlation threshold (μ): 0.0  
- Patience (λ): 3–5 epochs  

CSDC replaces:
- early stopping  
- fixed max epoch training  
- validation-only stopping

---

# 🔧 Full Training Pipeline Overview

During training:
Load image ↓ Standard augmentations (crop, flip) ↓ Mixup or CutMix ↓ Forward pass → logits ↓ Compute label smoothing loss ↓ Backward pass (SAM first_step + second_step) ↓ Update weights

After every epoch:

Record training loss Record validation loss Compute rolling Pearson correlation If correlation < μ for λ epochs → STOP (CSDC) Select best validation model

---

# ⚙️ Hyperparameters Summary

| Component | Value |
|----------|--------|
| Mixup α | 0.2 |
| CutMix probability | 0.5 |
| Label smoothing | 0.1 |
| Adaptive Dropout | 0.2 |
| Optimizer | SAM + AdamW |
| Learning rate | 0.001 |
| Weight decay | 1e-4 |
| LR scheduler | Cosine Annealing |
| CSDC window size (ω) | 10–20 |
| CSDC threshold (μ) | 0.0 |
| CSDC patience (λ) | 3–5 |

---

# 📦 Supported Datasets

- CIFAR-10  
- Caltech-101  
- TinyImageNet  

Each dataset uses the **same regularization framework** with small tuning.

---

# 🎉 Summary

This framework combines **robust training regularizers** with a **modern stopping method (CSDC)**:

- **Input-level** → Mixup, CutMix  
- **Model-level** → Adaptive Dropout  
- **Loss-level** → Label Smoothing  
- **Optimizer-level** → SAM + AdamW  
- **Stopping-level** → CSDC  

Together, these techniques create a clean, powerful, and reproducible system for preventing overfitting in CNNs.