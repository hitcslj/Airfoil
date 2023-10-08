import os
import numpy as np

root_path = 'data/airfoil/picked_uiuc'
allData = []
## 使用os.walk()函数遍历文件夹
for root, dirs, files in os.walk(root_path):
    for file in files:
        file_path = os.path.join(root, file)
        # Do something with the file_path
        allData.append(file_path)

xx,yy = [],[]
for i, file_path in enumerate(allData):
    with open(file_path, 'r') as f:
        lines = f.readlines()
        x = []
        y = []
        for line in lines:
            point = line.strip().split()
            x.append(float(point[0]))
            y.append(float(point[1]))
        xx.append(x)
        yy.append(y)

## 从数据中可以看出来，采样的所有飞机的x坐标是一样的，y坐标是不一样的
for i in range(1,len(xx)):
    if xx[i]!=xx[i-1]:
        print('False')
x = xx[0]

ind = np.where(np.array(x)==0)[0][0]
print(ind) # 100,中间位置
up = x[:ind]
low = x[ind+1:][::-1]
print(len(up),len(low))
# 从数据中可以看出来，采样的飞机上表面和下表面的x坐标是一样的（有几个不相等但是因为计算误差1e-7）
for a,b in zip(up,low):
    if a!=b:
      print(a,b)


