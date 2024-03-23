# 使用说明：



## 模型的总体预览图

![总览图](./source/model_overview.jpg)

![source/model_framework.png](./source/model_framework.png)

## 环境配置

> pip install -r requirements.txt


## 基础的encoder - decoder

输入：N个点 (N,2)

输出：N个点 (N,2)

> python train.py # 训练模型


> python infer.py # 测试模型


## 基础的基于物理量重建


输入：n （n = N/10）个关键点 + 8个物理量  {'input':input,'output':data,'params':params}

输出：N个点 (N,2)

> python train_parsec.py # 训练模型

> python infer_parsec.py # 测试模型

## 基础的基于物理量的编辑

> python train_editing

> python infer_editing



## 实验精度记录

200个点，l2损失: 2.5987829166718574e-07(30000ep)

sample 20个点，l2损失:1.485746088780964e-4(30000ep)

20个点+10个物理量，l2损失: 3.921557338764113e-06 （10000ep）

20个点+10个物理量+引入position embedding，l2损失: 4.803270279751077e-06 (20000ep)

20个点+10个物理量+引入position embedding+中弧线，l2损失: 




