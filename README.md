# Capsular Contracture Classification Model

## Introduction
Capsular contracture is a condition that can happen after breast implant surgery. It occurs when scar tissue around the implant becomes hard and tight. This can cause pain, discomfort, or changes in how the breast looks or feels. Early detection of this problem can help doctors treat it faster and avoid complications.

![Removed_Breast_Implant_ _Capsular_Contracture_Tissue webp](https://github.com/user-attachments/assets/4e7dcd2e-59af-4e24-bf2d-aede74e20848)

#### Image taken after a capsular contracture scar tissue removal surgery, showing the removed implant and removed tissue capsule

## Problem
Doctors usually look at 3D medical images, like MRI scans, to find signs of capsular contracture. However, analyzing these images takes time and can sometimes lead to mistakes. This project aims to use a machine learning model to help identify this condition quickly and accurately.

## Goal
The goal of this project is to:
- Analyze 3D breast MRI scans in `.nii.gz` format.
- Detect if there is excess tissue (a sign of capsular contracture).
- Provide a tool to help doctors diagnose the condition more easily.

## Libraries and Tools
The following libraries are used:
- **PyTorch:** For building and training the deep learning model.
- **MONAI:** For medical image data processing and transformations.
- **NumPy:** For handling converted data in `.npy` format.
- **Matplotlib:** For visualizing data distributions and results.
- **Scikit-learn:** For calculating evaluation metrics.


## How the Model Works

### 1. Data Preparation
- **Data Conversion:** The original MRI scans in NIfTI format (`.nii.gz`) were converted into NumPy arrays (`.npy`) for faster processing and easier integration with deep learning models.
- **CSV File Creation:** A CSV file was created to organize the dataset, where each row contains:
  - The file path of the corresponding NumPy array.
  - The classification label.
- **Image Loading:** The MONAI library is used to load NumPy arrays during training and testing.
- **Preprocessing:** Preprocessing steps include:
  - Ensuring channel-first formatting for compatibility with PyTorch.
  - Scaling intensity values to normalize image brightness.
  - Resizing images to ensure uniform dimensions.
- **Data Augmentation:** Data augmentation techniques, such as rotations, flips, and intensity adjustments, are applied to enhance model robustness.

### 2. Model Selection
- The project uses a **ResNet-18** architecture, which is modified to handle 3D image inputs.
- The ResNet architecture is chosen for its efficiency in extracting features while avoiding issues like vanishing gradients, thanks to its residual connections.

### 3. Training Process
- **Loss Function:** Binary cross-entropy loss is used for optimizing the classification task.
- **Optimizer:** The Adam optimizer ensures efficient gradient updates.
- **Metrics:** Key evaluation metrics include accuracy, precision, recall, F1-score, and AUC-ROC.

