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

在项目的根文件夹下:
```bash
# interpolate airfoil to specified number of points
python dataload/interpolate.py 

# split train/val/test
python dataload/datasplit.py 

# generate parsec feature
python dataload/parsec_direct.py 
```


## Usage

在项目的根文件夹下:

reproduce baseline: [softvae](https://arxiv.org/abs/2205.02458), [cvae-gan](https://www.sciencedirect.com/science/article/pii/S1000936121000662)

```bash
# train soft-cvae condition on keypoint&parsec
python train_soft_vae.py 

# eval editing performance
python eval_editing_softvae.py

# train cvae-gan condition on keypoint&parsec
python train_cvae_gan.py 

# eval editing performance
python eval_editing_cvae_gan.py

```


```bash
# train pkvae condition on keypoint&parsec
python train_pk_vae.py

# eval editing performance
python eval_editing_pkvae.py

# train editing keypoint condition on source_keypoint,target_keypoint,source_param, generate: target_param
python train_ekvae.py

# join train editing keypoint & pkvae condition source_keypoint,target_keypoint,source_param, generate: target_point
python train_ek_pkvae.py

# train editing param condition on source_param,target_param,source_keypoint, generate: target_keypoint
python train_epvae.py

# train editing param condition on source_param,target_param,source_keypoint, generate: target_point
python train_ep_pkvae.py

```
