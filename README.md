# A Texture Reconstructive Downsampling for Multi-Scale Object Detection in UAV Remote Sensing Images
The pytorch implementation of TRD: 
## Introduction

**Abstract**:



## Installation and Get Started

Required environments:
* Linux
* Python 3.6+
* PyTorch 1.3+
* CUDA 9.2+
* GCC 5+
* [MMCV](https://mmcv.readthedocs.io/en/latest/#installation)
* [cocoapi-aitod](https://github.com/jwwangchn/cocoapi-aitod)


Install:

Note that this repository is based on the [MMYOLO]([https://github.com/open-mmlab/mmdetection](https://github.com/open-mmlab/mmyolo)). Assume that your environment has satisfied the above requirements, please follow the following steps for installation.

```shell script
conda create -n mmyolo python=3.8 pytorch==1.10.1 torchvision==0.11.2 cudatoolkit=11.3 -c pytorch -y
conda activate mmyolo
pip install openmim
mim install "mmengine>=0.6.0"
mim install "mmcv>=2.0.0rc4,<2.1.0"
mim install "mmdet>=3.0.0,<4.0.0"
git clone https://github.com/open-mmlab/mmyolo.git
cd mmyolo
# Install albumentations
pip install -r requirements/albu.txt
# Install MMYOLO
mim install -v -e .
```
## Main Results
Table 1. **Training Set:** VisDrone-DET trainval set, **Validation Set:** VisDrone-DET val set, 300 epochs. [VisDrone-DET]([https://github.com/jwwangchn/AI-TOD](https://github.com/VisDrone/VisDrone-Dataset))

## Visualization
The images are from the VisDrone2019 datasets. 

## Citation
If you find this work helpful, please consider citing:


