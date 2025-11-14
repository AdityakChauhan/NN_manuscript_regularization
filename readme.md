# 📘 Regularization Framework for CNN Training

This repository provides a **strong yet simple regularization framework** for training Convolutional Neural Networks (CNNs) on image classification datasets such as **CIFAR-10**, **TinyImageNet**, and **Caltech-101**.

The goal is to **prevent overfitting** and ensure stable generalization while keeping the method **clean, reproducible, and easy to understand**.

---

# 🚀 Regularization Techniques Used

### **1. Mixup**
Smoothly blends images and labels to prevent memorization.

### **2. CutMix**
Pastes random patches between images to improve localization robustness.

### **3. Label Smoothing**
Reduces overconfidence and improves generalization.

### **4. Adaptive Dropout**
Dropout layer where the probability can adjust based on layer depth or difficulty.  
Adds noise-based regularization and prevents co-adaptation of neurons.

### **5. CSDC (Cross-Scale Distillation Consistency)**
Encourages the model to learn **scale-invariant features** by matching representations between:
- the original image  
- a resized (scaled) version  

Acts as a strong representation-level regularizer.

### **6. SAM + AdamW Optimizer**
- SAM finds flat minima → improved generalization  
- AdamW adds weight decay stability  

### **7. Standard Augmentations**
- RandomResizedCrop  
- RandomHorizontalFlip  
- Normalize  

---

# 🔧 Training Pipeline Overview

Below is the exact sequence of steps during training:

