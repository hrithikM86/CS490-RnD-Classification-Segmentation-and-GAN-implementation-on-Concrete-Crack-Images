# Classification, Segmentation, and GAN Implementation on Concrete Crack Images

**Hrithik Mhatre** and **Vamshika Sutar**  
**Supervisors**: Prof. Abir De (Dept. of CSE, IIT Bombay) & Prof. Alankar Alankar (CMInDS, IIT Bombay)  
**Department**: Civil Engineering, IIT Bombay  

---

## 🧠 Introduction

Early detection of concrete cracks is vital for structural integrity and safety. Traditional inspection methods are error-prone and inefficient. Our work proposes a deep learning-based framework that combines:

- **GANs** for minority class augmentation  
- **ResNet50** for image-level classification  
- **U-Net** for pixel-wise crack segmentation  

This pipeline enhances model robustness in data-scarce environments and enables automated structural health monitoring.

---

## 🧪 Generative Adversarial Networks (GANs)

To mitigate data imbalance, we trained a GAN to generate realistic cracked images. These were combined with real samples to improve classifier generalization.

- **Architecture**: DCGAN with ConvTranspose2D (Generator) and Conv2D (Discriminator)
- **Training**: 60 epochs using Adam optimizer (lr = 0.0002, β₁ = 0.5)
- **Input**: Latent vector `z` of size 100
- **Output**: 64×64×3 RGB synthetic images
- **Losses**: Binary cross-entropy for both Generator and Discriminator

---

## 🧩 Concrete Crack Classification with ResNet50

Using the original and GAN-augmented dataset (227×227 RGB JPEGs):

- **Base Model**: Pre-trained ResNet50 (`include_top=False`)
- **Classifier Head**: 2 Dense layers + Softmax (binary classification)
- **Training**:
  - Data generators with augmentation
  - Batch size: 64 | Input size: 100×100 | Optimizer: Adam
  - EarlyStopping with validation loss monitoring
- **Performance**:
  - External Test 1: Accuracy = 99.33%, F1 = 0.99, ROC AUC = 0.99
  - External Test 2: Accuracy = 82.02%, F1 = 0.77, ROC AUC = 0.59

---

## 🧭 Crack Segmentation with U-Net

We implemented U-Net to achieve pixel-level segmentation of cracks.

- **Input**: 240×160 RGB images (GAN + real)
- **Preprocessing**: Normalized [0, 1], PNG masks binarized
- **Training**:
  - Batch size: 8 | Epochs: 100 | Patience: 30
  - Optimizer: Adam | Loss: Dice coefficient loss
- **Evaluation**:
  - Metrics: Dice Score, Mean IoU
  - Segmentation visualizations confirm accurate localization of cracks

---

## ✅ Conclusion

This work demonstrates a robust, end-to-end pipeline for concrete crack analysis:

- **GANs** effectively overcome data scarcity
- **ResNet50** achieves strong generalization in classification
- **U-Net** accurately segments cracks at pixel-level

Our approach lays the groundwork for scalable, automated structural health monitoring in real-world infrastructure systems.

---
