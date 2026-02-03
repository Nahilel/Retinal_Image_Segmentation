# STD-Net: Structure-Texture Demixing Network for Retinal Image Segmentation

## 📌 Overview
This repository contains a full, from-scratch implementation of **STD-Net**, a bimodal deep learning architecture designed for high-precision retinal image segmentation. Retinal images are notoriously difficult to segment due to the complex mix of anatomical structures (vessels, optic disc) and fine textures or noise. 

Our approach solves this by explicitly **demixing** the input image into its structural and textural components before performing segmentation. By isolating the "signal" (structure) from the "noise" (texture), the model achieves robust performance even on low-contrast datasets.

---

## 🏗️ Architecture Design
The network is composed of two primary modules that work in a unified pipeline:

### 1. Structure-Texture Demixing (STD) Module
Based on the principle that an image $I$ is the sum of a structure $S$ and a texture $T$ ($I = S + T$):
* **Texture Extraction**: 10 convolutional layers (64 filters each) with LeakyReLU activations ($\alpha=0.01$) to isolate the textural component.
* **Texture Block**: A refinement stage that extracts residual structural information from the texture using **Adaptive Normalization** and $1 \times 1$ convolutions.

### 2. M-Net Segmentation Backbone
The extracted structure is fed into an **M-Net** architecture:
* **Multi-scale Input**: The structure is down-sampled and fed into multiple levels of the encoder.
* **Deep Supervision**: The decoder produces grayscale outputs at every level, which are up-sampled and fused to create the final segmentation mask.

---

## 🧪 Mathematical Framework
The model is optimized using a composite loss function that enforces both demixing quality and segmentation accuracy:

$$\mathcal{L}_{total}(S,T,R) = \mathcal{L}_{seg}(R) + \mu(\mathcal{L}_{t}(T) + \lambda\mathcal{L}_{s}(S))$$

* **Segmentation Loss ($\mathcal{L}_{seg}$)**: A weighted Cross-Entropy loss across multiple output scales:
    $$L_{seg} = \frac{1.0L_{final} + 0.3L_{seg2} + 0.1L_{seg3} + 0.05L_{seg4}}{1.0 + 0.3 + 0.1 + 0.05}$$
* **Texture Loss ($\mathcal{L}_{t}$)**: $L_1$ norm to encourage sparsity.
* **Structure Loss ($\mathcal{L}_{s}$)**: Total Variation (TV) loss to ensure spatial smoothness:
    $$\mathcal{L}_{s}(S) = \sum_{i,j} \sqrt{(S_{i+1,j}-S_{i,j})^{2} + (S_{i,j+1}-S_{i,j})^{2}}$$

---

## 🛠️ Implementation Details
### Preprocessing Pipeline
To handle the variability in medical imaging, we implemented specialized preprocessing for different tasks:
* **Vessel Segmentation (DRIVE)**: Images are resized to $512 \times 512$ and processed with **CLAHE** (Contrast Limited Adaptive Histogram Equalization) to enhance fine vessels.
* **Disc/Cup Segmentation (ORIGA/REFUGE)**: Images are converted to **Polar Coordinates** to linearize the radial structure of the optic disc and cup.

### Training Configuration
* **Optimizer**: Adam with an initial learning rate of $0.001$.
* **Augmentation**: Horizontal/vertical flips and $\pm 15^\circ$ rotations.
* **Environment**: PyTorch with GPU acceleration (CUDA).

---

## 📊 Performance & Results
The model was evaluated on three major datasets. While our implementation remains competitive, slight variances from the original paper are attributed to specific hardware constraints and differences in ROI cropping strategies.

### Quantitative Metrics (DRIVE Dataset)
| Acc | AUC | Sen | Spe | IOU |
| :--- | :--- | :--- | :--- | :--- |
| **0.9448** | **0.9663** | **0.7163** | **0.9781** | **0.6228** |

---

## 🚀 Getting Started
### Dependencies
* Python 3.x
* PyTorch / Torchvision
* OpenCV
* NumPy / Pandas
* Matplotlib / tqdm

### Usage
The complete pipeline is available in the `retinal_segmentation.ipynb` notebook. It includes:
1.  Data loading and preprocessing scripts.
2.  The full `STDNetFullModel` class.
3.  The multi-scale training loop with validation monitoring.

---

## 📜 Authors
Developed by **Nahil El Bezzari, Yassine Lazizi, Pascal Le, Anis Melaimi, and Nadir Nehili**.

---
*Date: December 2, 2025*
