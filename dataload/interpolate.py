# 对uiuc的数据进行插值处理， 采样出257个点
import os
import numpy as np
from scipy.interpolate import splev,splprep,splrep
from parsec_direct import Fit_airfoil

uiuc_path = 'data/airfoil/picked_uiuc'
interpolated_path = 'data/airfoil/interpolated_uiuc'
os.makedirs(interpolated_path, exist_ok=True)

import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import splev, splrep,splprep


def interpolote_up(data, x_coords):
    x = data[:,0][::-1]
    y = data[:,1][::-1]
    spl = splrep(x, y, s = 0)
    y_interp = splev(x_coords, spl)
    x_coords = x_coords[::-1]
    y_interp = y_interp[::-1]
    return np.array([x_coords, y_interp]).T

def interpolote_down(data, x_coords):
    x = data[:,0]
    y = data[:,1]
    spl = splrep(x, y, s = 0)
    y_interp = splev(x_coords, spl)
    return np.array([x_coords, y_interp]).T


def interpolate(data,s_x = 100, t_x = 129):
    # bspline 插值
    up = data[:s_x]  # 上表面原来为100个点, 插值为128个点
    mid = data[s_x:s_x+1]
    down = data[s_x+1:]  # 下表面原来为100个点，插值为128个点

    theta = np.linspace(np.pi, 2*np.pi, t_x)
    x_coords = (np.cos(theta) + 1.0) / 2
    x_coords = x_coords[1:]  # 从小到大
    
    up_interp = interpolote_up(up, x_coords)

    down_interp = interpolote_down(down, x_coords)

    # 组合上下表面
    interpolated_data = np.concatenate((up_interp, mid, down_interp))
    

    # x,y = data[:,0],data[:,1]
    # x2,y2 = interpolated_data[:,0],interpolated_data[:,1]

    # # 可视化验证
    # plt.plot(x, y, 'o', x2, y2)
    # plt.show()
    # plt.savefig('interpolated.png')

    return interpolated_data

for root,dir,files in os.walk(uiuc_path):
    for file in files:
        data = []
        with open(os.path.join(root,file),'r') as f:
          # 逐行读取文件内容
          for line in f:
              # 移除行尾的换行符，并将每行拆分为两个数值
              values = line.strip().split()
              # 将数值转换为浮点数，并添加到数据列表中
              data.append([float(values[0]), float(values[1])])
          # 对 data进行插值
          data = np.array(data)
          interpolated_data = interpolate(data)
        # 保存插值后的数据
        with open(os.path.join(interpolated_path,file),'w') as f:
            for x,y in interpolated_data:
                f.write(f'{x} {y}\n')
print('----finished----')