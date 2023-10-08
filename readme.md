## 使用说明：

## 基础的encoder - decoder

输入：N个点 (N*2)
输出：N个点 (N*2)

> python train.py # 训练模型


> python infer.py # 测试模型


## 基础的基于物理量重建

n = N/10
输入：n 个关键点 + 8个物理量  {'input':input,'output':data,'params':params}
输出：N个点 (N*2)

> python train_parsec.py # 训练模型

> python infer_parsec.py # 测试模型

