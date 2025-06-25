# CACKD

CACKD is a [segmentation_models_pytorch](https://github.com/qubvel/segmentation_models.pytorch)-based framework that leverages Covariance Attention and Correlation Information Fusion for Knowledge Distillation, boosting semantic segmentation performance on lens flare–degraded road scene images.

---

## 🎯 Key Features

- **smp-based**: Plug-and-play support for UNet, DeepLabV3+, etc.  
- **Channel Covariance Attention Module (CCAM)**: Captures second-order channel statistics to highlight robust features under varying flare conditions.  
- **Class-wise Cross-Correlation KD**: Distills a full C×C correlation matrix of class activations, preserving inter-class relationships during training.  
- **Zero Inference Overhead**: All attention and distillation modules are active only during training; the Student model remains lightweight at inference.  
- **Synthetic Flare Dataset Compatibility**: Includes data loaders and augmentations tailored for Syn-flare CamVid and Syn-flare KITTI benchmarks.  
- **Built-in Grad-CAM Tools**: Multi-layer Grad-CAM scripts for visualizing how CCAM adjusts focus under flare—ideal for debugging and paper figures.

---

## 📦 Installation

```bash
pip install cackd

Or from source:

git clone https://github.com/yourusername/CACKD.git
cd CACKD
pip install -e .

## 🗂 Datasets (Syn-flare CamVid & KITTI)

### Syn-flare CamVid
```bash
gdown --id 1-XXXXXXXXXXXXXXX -O syn-flare-camvid.zip
```

### Syn-flare KITTI
```bash
gdown --id 1-YYYYYYYYYYYYYYY -O syn-flare-kitti.zip
```


