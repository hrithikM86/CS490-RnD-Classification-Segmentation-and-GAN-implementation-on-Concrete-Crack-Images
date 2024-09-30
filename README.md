# CS 490: Research and Development Project
**Project Title:** Classification, Segmentation, and GAN Implementation on Concrete Crack Images  
**Student:** Hrithik Mhatre (Roll No. 210040092)  
**Supervisors:** Prof. Abir De and Prof. Alankar Alankar  
**Department:** Civil Engineering, IIT Bombay  
**Date:** April, 2024  
**Github:** [CS-490-RND Repository](https://github.com/hrithikM86/CS-490-RND)

---

## Table of Contents
1. [Introduction](#introduction)
2. [Concrete Crack Detection Utilizing ResNet50](#concrete-crack-detection-utilizing-resnet50)
   - [Dataset](#dataset)
   - [Data Generator and Image Preprocessing](#data-generator-and-image-preprocessing)
   - [Model Creation](#model-creation)
   - [Training the Model](#training-the-model)
   - [Results](#results)
     - [Performance Evaluation on the Same Dataset](#performance-evaluation-on-the-same-dataset)
     - [Evaluating Model Performance on a New Dataset Using Pre-Trained Weights](#evaluating-model-performance-on-a-new-dataset-using-pre-trained-weights)
3. [Generative Adversarial Network (GAN)](#generative-adversarial-network-gan)
   - [Introduction to GAN](#introduction-to-gan)
   - [Dataset](#dataset-gan)
   - [Hyper Parameters](#hyper-parameters)
   - [Image Preprocessing](#image-preprocessing)
   - [Data Loader](#data-loader)
   - [Weights](#weights)
   - [Generator Architecture](#generator-architecture)
   - [Discriminator Architecture](#discriminator-architecture)
   - [Training Loop](#training-loop)
   - [Results](#results-gan)

---

## Introduction
In civil engineering, detecting concrete cracks is essential for maintaining structural integrity. This project aims to develop robust algorithms for crack classification, segmentation, and data generation using Generative Adversarial Networks (GANs) on concrete crack images. By integrating these techniques, the goal is to improve the detection and segmentation of diverse crack types in various environmental conditions.

---

## Concrete Crack Detection Utilizing ResNet50

### Dataset
- **Source:** [Oluwaseunad's dataset](https://www.kaggle.com/oluwaseunad/concrete-and-pavement-crack-images)
- **Total Images:** 30,000 (categorized into cracked and non-cracked)
- **Image Dimensions:** 227 x 227 pixels, RGB JPEG

### Data Generator and Image Preprocessing
- **Data Augmentation** was applied to training data to improve model robustness.
- **Preprocessing**: `preprocess_input` from TensorFlow's ResNet50 was used for standardization.
- **Validation Split**: 20% of training data used for validation.
- **Batch Size**: 64 images per batch.
- **Image Size**: Resized to 100 x 100 pixels.

### Model Creation
- Pre-trained **ResNet50** architecture with ImageNet weights.
- Custom classification layers added for crack detection.
- **Optimizer**: Adam
- **Loss Function**: Categorical Cross-Entropy
- Early stopping implemented to prevent overfitting.

### Training the Model
- Trained for **100 epochs** with early stopping on validation loss.

### Results

#### Performance Evaluation on the Same Dataset
- **Test Accuracy**: 99.33%  
- **F1 Score**: 0.99  
- **ROC AUC**: 0.99  

#### Evaluating Model Performance on a New Dataset Using Pre-Trained Weights
- **Dataset 1**: [hesighsrikar/concrete-crack-images-for-classification](https://www.kaggle.com/hesighsrikar/concrete-crack-images-for-classification)
  - Test Accuracy: 82.02%  
  - F1 Score: 0.77  
  - ROC AUC: 0.59  
  - **Note:** Mislabelling present in dataset led to lower performance.

---

## Generative Adversarial Network (GAN)

### Introduction to GAN
- GANs consist of two neural networks: the **Generator** (to create images) and the **Discriminator** (to distinguish between real and fake images).
- Used for generating synthetic cracked concrete images.

### Dataset
- **Source**: [thesighsrikar/concrete-crack-images-for-classification](https://www.kaggle.com/thesighsrikar/concrete-crack-images-for-classification)
- **Images Used**: 10,000 cracked concrete images (227 x 227 pixels, RGB).

### Hyper Parameters
- **Batch Size**: 64  
- **Image Size**: 64 x 64  
- **Latent Vector Size**: 100  
- **Epochs**: 60  
- **Learning Rate**: 0.0002  
- **Optimizer**: Adam (beta1 = 0.5)

### Image Preprocessing
- Resized and normalized using mean (0.5, 0.5, 0.5) and standard deviation (0.5, 0.5, 0.5).

### Data Loader
- **torch.utils.data.DataLoader** used with batch size of 64 and shuffle enabled.

### Weights
- **Weights Initialization**: Convolutional layers initialized from a normal distribution with mean 0 and std 0.02.

### Generator Architecture
- **ConvTranspose2D** layers for upsampling.
- **Batch Normalization** and **ReLU** activation for stabilizing the network.
- Final layer uses **Tanh** activation to generate pixel values in the range [-1, 1].

### Discriminator Architecture
- **Conv2D** layers for feature extraction.
- **LeakyReLU** for non-linearity.
- Outputs single-channel binary predictions (real/fake).

### Training Loop
- GAN was trained for **60 epochs** with alternating updates to the Generator and Discriminator networks.

### Results
- Generated synthetic images of cracked concrete closely resembling real-world cracks.

---

## Conclusion
This project successfully implemented classification, segmentation, and GAN techniques for concrete crack detection and image generation. The results demonstrate high accuracy in classification and effective image generation using GANs, making this approach suitable for real-world crack detection and augmentation of crack datasets.

---
