import os
import matplotlib.pyplot as plt
plt.switch_backend('agg')
import numpy as np
from scipy.spatial.distance import pdist, squareform

def get_data(path):
  with open(path) as f:
      data = np.array([list(map(float,line.strip().split())) for line in f.readlines()])
  return data

def getFilePath(root_path):
    file_paths = []
    for root, dirs, files in os.walk(root_path):
        for file in files:
            file_path = os.path.join(root, file)
            file_paths.append(file_path)
    return file_paths


def vis_airfoil2(source,target,idx,dir_name='output_airfoil',sample_type='ddpm'):
    os.makedirs(dir_name,exist_ok=True)
 
    ## 将source和target放到一张图
    plt.scatter(source[:, 0], source[:, 1], c='red', label='source',s=0.5)  # plot source points in red
    plt.scatter(target[:, 0], target[:, 1], c='blue', label='target',s=0.5)  # plot target points in blue
    plt.legend()  # show legend

    file_path = f'{dir_name}/{sample_type}_{idx}.png'
    plt.savefig(file_path, dpi=100, bbox_inches='tight', pad_inches=0.0)
    plt.clf()

def vis_airfoil3(source,target_pred,target,idx,dir_name='output_airfoil',sample_type='ddpm'):
    os.makedirs(dir_name,exist_ok=True)
 
    ## 将source和target放到一张图
    plt.scatter(source[:, 0], source[:, 1], c='red', label='source',s=0.5)  # plot source points in red
    plt.scatter(target_pred[:, 0], target_pred[:, 1], c='green', label='target_pred',s=0.5)  #  plot target pred points in blue
    plt.scatter(target[:, 0], target[:, 1], c='blue', label='target',s=0.5)  # plot target points in blue
    plt.legend()  # show legend

    file_path = f'{dir_name}/{sample_type}_{idx}.png'
    plt.savefig(file_path, dpi=100, bbox_inches='tight', pad_inches=0.0)
    plt.clf()



def calculate_smoothness(airfoil):
    smoothness = 0.0
    num_points = airfoil.shape[0]

    for i in range(num_points):
        p_idx = (i - 1) % num_points
        q_idx = (i + 1) % num_points

        p = airfoil[p_idx]
        q = airfoil[q_idx]

        if p[0] == q[0]:  # 处理垂直于x轴的线段
            distance = abs(airfoil[i, 0] - p[0])
        else:
            m = (q[1] - p[1]) / (q[0] - p[0])
            b = p[1] - m * p[0]

            distance = abs(m * airfoil[i, 0] - airfoil[i, 1] + b) / np.sqrt(m**2 + 1)

        smoothness += distance

    return smoothness


def cal_diversity_score(data, subset_size=10, sample_times=1000):
    # Average log determinant
    N = data.shape[0]
    data = data.reshape(N, -1) 
    mean_logdet = 0
    for i in range(sample_times):
        ind = np.random.choice(N, size=subset_size, replace=False)
        subset = data[ind]
        D = squareform(pdist(subset, 'euclidean'))
        S = np.exp(-0.5*np.square(D))
        (sign, logdet) = np.linalg.slogdet(S)
        mean_logdet += logdet
    return mean_logdet/sample_times



if __name__ == '__main__':
  # smoothness_value 计算
  # airfoil = get_data('data/airfoil/interpolated_uiuc/2032c.dat')

  # smoothness_value = calculate_smoothness(airfoil)
  # print("Smoothness:", smoothness_value)
   
  # 示例 diversity score计算, 只选取前缘半径作为一个点进行计算
  # 先计算一下数据集的diversity 
  file_paths = getFilePath('data/airfoil/supercritical_airfoil')
  data = []
  for file_path in file_paths:
    data.append(get_data(file_path))
  data = np.array(data)
  s = cal_diversity_score(data[:10])
  print("Diversity score:", s)
  
  # label error 计算
  