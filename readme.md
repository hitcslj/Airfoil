# Airfoil-training

Editing keypoint &amp; parameters, built upon [pytorch](https://pytorch.org/)



## Installation

```bash
conda create --name airfoil python=3.8
conda activate airfoil
pip install -r requirements.txt
```

## Dataset

请将数据集存放在 `data` 文件夹下, 默认数据集为 `data/airfoil/supercritical_airfoil/*.dat`

在项目的dataload文件夹下:
```bash
# split train/val/test
python datasplit.py 

# generate parsec feature
python parsec_direct.py 
```


## Usage

在项目的根文件夹下:

```bash
# train cvae condition on keypoint&parsec
python train_cvae.py

# train editing parsec: source_param,target_param,source_keypoint -> target_keypoint 
python train_editing_parsec.py

# join train editing parsec&cvae: source_param,target_param,source_keypoint -> target_point
python train_editing_parsec_recons.py

# train editing keypoint: source_keypoint,target_keypoint,source_param -> target_param
python train_editing_keypoint.py

# train editing keypoint &cvae: source_keypoint,target_keypoint,source_param -> target_point
python train_editing_keypoint_recons.py

# refinement test
python infer_editing_parsec_refine.py
```
