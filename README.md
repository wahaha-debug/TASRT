<div align="center"> 

## Enhancing RGB-Thermal Semantic Segmentation with Text-Guided Sparse Mixture-of-Experts</div>

</div>

## 💬 Introduction

RGB-Thermal (RGB-T) semantic segmentation leverages the complementary cues from visible and thermal imagery to achieve reliable scene understanding under challenging conditions such as low light and adverse weather. However, existing methods often rely on fixed fusion schemes, neglecting the dynamic reliability of modalities and text semantics. We introduce TASRT, a text-guided RGB-T segmentation framework that integrates adaptive Mixture-of-Experts (MoE) fusion. TASRT incorporates three synergistic components: Dual Token-Guided Fusion (D-TGF) for semantically interpretable representations, Deformable Structure-Conditioned Cross-modal Attention (DSCSE) for robust alignment, and vision-driven sparse MoE blocks for adaptive, efficient fusion. Experiments on MFNet, PST900, and FMB datasets demonstrate superior segmentation accuracy (e.g., achieving 89.16\% mIoU on PST900) and competitive efficiency, supporting MoE-based adaptive fusion as a principled approach to multimodal perception.

## 🚀 Updates
- [x] 12/2025: init repository and release the code.
- [x] 12/2025: release TASRT model weights. Download from [**GoogleDrive**](https://drive.google.com/drive/folders/18Y0xnkEDwxTEgFzhnIkKCGemVlfprs67?dmr=1&ec=wgc-drive-globalnav-goto).

## 🔍 Environment

First, create and activate the environment using the following commands: 
```bash
conda env create -f environment.yaml
conda activate TASRT
```

## 📦 Data preparation
Download the dataset:
- [PST900](https://github.com/haqishen/MFNet-pytorch), for PST900 dataset with RGB-Infrared modalities
- [FMB](https://github.com/JinyuanLiu-CV/SegMiF), for FMB dataset with RGB-Infrared modalities.
- [MFNet](https://github.com/haqishen/MFNet-pytorch), for MFNet dataset with RGB-Infrared modalities.

Then, put the dataset under `data` directory as follows:

```
data/
├── PST900
│   ├── test
│   │   ├── thermal
│   │   ├── labels
│   │   └── rgb
│   ├── train
│   │   ├── thermal
│   │   ├── labels
│   │   └── rgb
```

## 📦 Model Zoo


### PST900
| Model-Modal      | mIoU   | weigh |
| :--------------- | :----- | :----- |
| PST900      | 89.16 | [GoogleDrive](https://drive.google.com/drive/folders/18Y0xnkEDwxTEgFzhnIkKCGemVlfprs67?dmr=1&ec=wgc-drive-globalnav-goto) |

## Training

Before training, please download pre-trained SAM, and put it in the correct directory following this structure:

```text
checkpoints
├── download_ckpts.sh
├── sam2_hiera_small.pth
├── sam2_hiera_tiny.pth
├── sam2_hiera_base_plus.pth
└── sam2_hiera_large.pth
```

To train TASRT model, please update the appropriate configuration file in `configs/` with appropriate paths and hyper-parameters. Then run as follows:

```bash
python -m tools.train_mm 
python -m tools.train_mm 
```

##  Evaluation

```text
python -m tools.val_mm2
```

