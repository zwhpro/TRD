# A Texture Reconstructive Downsampling for Multi-Scale Object Detection in UAV Remote Sensing Images
[Paper]()|[Code](https://github.com/zwhpro/TRD)
## Introduction

**Abstract**:
Unmanned aerial vehicle (UAV) remote sensing images present unique challenges to the object detection task due to uneven object densities, low resolution, and drastic scale variations. Downsampling is an important component of deep networks that expands the receptive field, reduces computational overhead, and aggregates features. However, object detectors using multi-layer downsampling result in varying degrees of texture feature loss for various scales in remote sensing images, degrading the performance of multi-scale object detection. To alleviate this problem, we propose a lightweight texture reconstructive downsampling module, called TRD. TRD models part of the texture features lost as residual information during downsampling. After modeling, cascading downsampling and upsampling operators provide residual feedback to guide the reconstruction of the desired feature map for each downsampling stage. TRD structurally optimizes the feature extraction capability of downsampling to provide sufficiently discriminative features for subsequent vision tasks. We replace the downsampling module of the existing backbone network with the TRD module and conduct a large number of experiments and ablation studies on a variety of remote sensing image datasets. Specifically, the proposed TRD module improves 3.1% AP over the baseline on the NWPU VHR-10 dataset. On the VisDrone-DET dataset, the TRD improves 3.2% AP over the baseline with little additional cost, especially the AP<sub>S</sub>, AP<sub>M</sub>, and AP<sub>L</sub> by 3.1%, 8.8%, and 13.9%, respectively. The results show that TRD enriches the feature information after downsampling and effectively improves the multi-scale object detection accuracy of UAV remote sensing images. The code is available at https://github.com/zwhpro/TRD.


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
conda create -n TRD python=3.8 -y
conda activate TRD

conda install cudatoolkit=11.6
conda install cudnn=8.4.1

conda install pytorch==1.12.1 torchvision==0.13.1 torchaudio==0.12.1 cudatoolkit=11.6 -c pytorch -c conda-forge

git clone https://github.com/zwhpro/TRD
cd TRD

pip install -U openmim -i
mim install -r requirements/mminstall.txt
mim install -r requirements/albu.txt
mim install -v -e .

```
## Core Idea
* We propose an inverse projection-based sample residual learning structure that computes the projection error to guide the reconstruction of the downsampled ideal feature maps by cascading the up-sampling and down-sampling operators.

* The proposed TRD method mitigates the inherent trade-off between resolution reduction and information loss. It effectively improves the multi-scale object detection accuracy in UAV remote sensing images.

* Extensive experiments are conducted with the TRD on VisDrone-DET and NWPU VHR-10 datasets, showing that TRD can improve the multi-scale detection performance of the model compared to the baseline with little additional cost.

## Train
**Training Set:** VisDrone-DET trainval set, **Validation Set:** VisDrone-DET val set, 300 epochs. [VisDrone-DET Link](https://github.com/VisDrone/VisDrone-Dataset)

YOLOv8-n w/ TRD training on VisDrone-DET (dataset path need to be changed in train.py):
```
python tools/train.py ./configs/TRD/yolov8_n_syncbn_fast_300e_visdrone_TRD.py
```

yolov5-s w/ TRD training on VisDrone-DET (dataset path need to be changed in train.py):
```
python tools/train.py ./configs/TRD/yolov5_s_syncbn_fast_300e_visdrone_TRD.py 
```

RTMDet tiny w/ TRD training on VisDrone-DET (dataset path need to be changed in train.py):
```
python tools/train.py ./configs/TRD/rtmdet_tiny_syncbn_fast_300e_nwpu_TRD.py
```

YOLOv8-n w/ [HWD/RFD/ADown/SCDown/FouriDown] training on VisDrone-DET (dataset path need to be changed in train.py):
```
python tools/train.py ./configs/TRD/yolov8_n_syncbn_fast_300e_visdrone_[HWD/RFD/ADown/SCDown/FouriDown].py
```

The NWPU VHR-10 dataset used in this study is accessible from http://pan.baidu.com/s/1hqwzXeG (accessed on 24 November 2022).
## Evaluation 
1.Detecting images with trained models:
```
python demo/image_demo.py \
./demo/images\
[path to config] \
[path to .pth]\
 --out-dir ./demo/result
 ```

2.Visualization ablationcam:
 ```
python demo/boxam_vis_demo.py \
./demo/images/0000001_02999_d_0000005.jpg \
[path to config] \
[path to .pth]\
--method ablationcam \
--target-layers neck.out_layers[0]\
--norm-in-bbox
 ```
3.Evaluation with trained models:
 ```
python tools/test.py \
[path to config] \
[path to .pth]
 ```
4.Model parameters and computational complexity:
 ```
python tools/analysis_tools/get_flops.py \
[path to config]
 ```
## Main Results
Table 1.Baseline is YOLOv8-n. **Training Set:** VisDrone-DET trainval set, **Validation Set:** VisDrone-DET val set, 300 epochs. [VisDrone-DET]([https://github.com/jwwangchn/AI-TOD](https://github.com/VisDrone/VisDrone-Dataset))
Method | Venue | AP | AP<sub>50</sub> | AP<sub>75</sub> | Param(M) | GFLOPs  
--- |:---:|:---:|:---:|:---:|:---:|:---:
[SC](https://github.com/ultralytics/ultralytics)      |   -      | 19.3  | 33.0  | 19.3  | 3.0   | 4.1 
[RFD](https://github.com/lwCVer/RFD)     | TGRS2023 | 20.2  | 34.6  | 20.0  | 3.5   | 6.3 
[HWD](https://github.com/apple1986/HWD)     | PR2023   | 18.4  | 32.1  | 18.1  | 2.8   | 3.8 
[SCDown](https://github.com/THU-MIG/yolov10)  | ARXIV2024| 19.5  | 33.4  | 19.2  | 2.7   | 3.8 
[FouriDown](https://github.com/zqcrafts/FouriDown)  | NeurIPS2024| 14.3  | 25.5  | 13.9  | 2.7   | 3.7 
[ADown](https://github.com/WongKinYiu/yolov9)  | ECCV2024 | 17.8  | 30.9  | 17.7  | 2.7   | 3.7 
**TRD** |      -   |**20.8**| **35.2**| **21.0**| 3.2 | 4.6 

All Pretrained Models: [BaiduNetdisk Link](https://pan.baidu.com/s/1k09fQvC09lcQBtu3FvhIZA?pwd=hfb2)  extraction code: hfb2

## Citation
If you find this work helpful, please consider citing:
```bibtex

```
Thanks to MMYOLO for the related work, which can be cited if the work needs to be considered:
```bibtex
@misc{mmyolo2022,
    title={{MMYOLO: OpenMMLab YOLO} series toolbox and benchmark},
    author={MMYOLO Contributors},
    howpublished = {\url{https://github.com/open-mmlab/mmyolo}},
    year={2022}
}
```
