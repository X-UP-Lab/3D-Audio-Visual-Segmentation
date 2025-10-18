# Installation Guide

## 1. Environment Setup

```bash
PROJECT_ROOT=$(pwd)
conda create -n avs3d python=3.10
conda activate avs3d
```

### 1.2. Install Core Dependencies

```bash
conda install pytorch=2.1 torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia -y
pip install .
```

***

## 2. Install Custom Submodules

### 2.1. Diff-Gaussian-Rasterization and Simple-KNN

```bash
cd $PROJECT_ROOT/src/avs_3d/echosegnet/audio_informed_gaussian_splatting/submodules
git clone --recursive https://github.com/ashawkey/diff-gaussian-rasterization
pip install ./diff-gaussian-rasterization
pip install ./simple-knn
cd $PROJECT_ROOT
```

### 2.2. Install CAOD (Class-Agnostic Open-set Detection)

**Clone the repository:**
```bash
git clone https://github.com/mmaaz60/mvits_for_class_agnostic_od.git $PROJECT_ROOT/src/avs_3d/echosegnet/owod_bind/submodules/caod
sed -i 's/from infer import Inference/from .infer import Inference/' $PROJECT_ROOT/src/avs_3d/echosegnet/owod_bind/submodules/caod/inference/minus_language.py
```

**Download Weights:**
The required file is **`MDef_DETR_minus_language_r101_epoch10.pth`**.
You should manually download this file from the following **folder link** and place it in the specified directory:

[Weights link for MDef_DETR_minus_language_r101_epoch10.pth](https://mbzuaiac-my.sharepoint.com/:f:/g/personal/muhammad_maaz_mbzuai_ac_ae/Erw0P4D7bDdKtl4BstL8XFsBv-k2W7Ya9rKOaZBIOrTEcQ?e=pL2VgF)

**Move to the checkpoint folder:**
```bash
mv /path/to/downloaded/MDef_DETR_minus_language_r101_epoch10.pth $PROJECT_ROOT/src/avs_3d/echosegnet/owod_bind/submodules/.checkpoints/
```

**Build and Export Path:**
```bash
cd $PROJECT_ROOT/src/avs_3d/echosegnet/owod_bind/submodules/caod/models/ops
sh make.sh

# Export PYTHONPATH to make the custom module importable. 
# This is temporary for the current session, do again for another session, otherwise:
#     from models.backbone import Backbone
# ModuleNotFoundError: No module named 'models'
export PYTHONPATH=$PYTHONPATH:$(realpath ../../)
cd $PROJECT_ROOT
```

### 2.2 Install ImageBind and Segment-Anything

```bash
CHECKPOINT_DIR=$PROJECT_ROOT/src/avs_3d/echosegnet/owod_bind/submodules/.checkpoints

# Install
pip install git+https://github.com/facebookresearch/ImageBind.git
pip install git+https://github.com/facebookresearch/segment-anything.git

# Download weights to the checkpoints folder
wget https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth -P $CHECKPOINT_DIR
```
