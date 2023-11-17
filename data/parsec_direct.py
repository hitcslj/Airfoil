from __future__ import division
from math import sqrt, tan, pi
import numpy as np
import matplotlib.pyplot as plt
from numpy.lib.shape_base import expand_dims
import pandas as pd
from scipy.optimize import minimize
from time import time
import os
from tqdm import tqdm
from scipy.interpolate import splev,splprep
from scipy import optimize

class fit_airfoil():
    '''
    Fit airfoil by 3 order Bspline and extract Parsec features.
    airfoil (npoints,2)
    '''
    def __init__(self,airfoil,iLE=128):
        self.iLE = iLE
        self.tck, self.u  = splprep(airfoil.T,s=0)

        # parsec features
        rle = self.get_rle()
        xup, yup, yxxup = self.get_up()
        xlo, ylo, yxxlo = self.get_lo()
        yteup = airfoil[0,1]
        ytelo = airfoil[-1,1]
        alphate, betate = self.get_te_angle()

        self.parsec_features = np.array([rle,xup,yup,yxxup,xlo,ylo,yxxlo,
                                         yteup,ytelo,alphate,betate]) 
        
        # 超临界翼型的特征
        xaft, yaft, yxxaft = self.get_aftload()
        # print(xaft, yaft, yxxaft)

    def get_rle(self):
        uLE = self.u[self.iLE]
        xu,yu = splev(uLE, self.tck,der=1) # dx/du
        xuu,yuu = splev(uLE, self.tck,der=2) # ddx/du^2
        K = (xu*yuu-xuu*yu)/(xu**2+yu**2)**1.5 # curvature
        return 1/K
    
    def get_up(self):
        def f(u_tmp):
            x_tmp,y_tmp = splev(u_tmp, self.tck)
            return -y_tmp
        
        res = optimize.minimize_scalar(f,bounds=(0,self.u[self.iLE]),tol=1e-10)
        uup = res.x
        xup ,yup = splev(uup, self.tck)

        xu,yu = splev(uup, self.tck, der=1) # dx/du
        xuu,yuu = splev(uup, self.tck, der=2) # ddx/du^2
        # yx = yu/xu
        yxx = (yuu*xu-xuu*yu)/xu**3
        return xup, yup, yxx

    def get_lo(self):
        def f(u_tmp):
            x_tmp,y_tmp = splev(u_tmp, self.tck)
            return y_tmp
        
        res = optimize.minimize_scalar(f,bounds=(self.u[self.iLE],1),tol=1e-10)
        ulo = res.x
        xlo ,ylo = splev(ulo, self.tck)

        xu,yu = splev(ulo, self.tck, der=1) # dx/du
        xuu,yuu = splev(ulo, self.tck, der=2) # ddx/du^2
        # yx = yu/xu
        yxx = (yuu*xu-xuu*yu)/xu**3
        return xlo, ylo, yxx

    def get_te_angle(self):
        xu,yu = splev(0, self.tck, der=1)
        yx = yu/xu
        alphate = np.arctan(yx)

        xu,yu = splev(1, self.tck, der=1)
        yx = yu/xu
        betate = np.arctan(yx)

        return alphate, betate
    
    # 后加载位置
    def get_aftload(self):
        def f(u_tmp):
            x_tmp,y_tmp = splev(u_tmp, self.tck)
            return -y_tmp
        
        res = optimize.minimize_scalar(f,bounds=(0.75,1),tol=1e-10)
        ulo = res.x
        xlo ,ylo = splev(ulo, self.tck)

        xu,yu = splev(ulo, self.tck, der=1) # dx/du
        xuu,yuu = splev(ulo, self.tck, der=2) # ddx/du^2
        # yx = yu/xu
        yxx = (yuu*xu-xuu*yu)/xu**3
        return xlo, ylo, yxx

if __name__ == '__main__':
  # Main code
  root_path = 'data/airfoil/picked_uiuc'
  allData = []
  ## 使用os.walk()函数遍历文件夹
  for root, dirs, files in os.walk(root_path):
      for file in files:
          file_path = os.path.join(root, file)
          # Do something with the file_path
          allData.append(file_path)
  allData = allData[:10]
  ## 翼型数据的横纵坐标
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
          x = x[:-1]
          y = y[:-1]
          xx.append(x)
          yy.append(y)
  y_coord_data = np.array(yy)
  x_coord_data = np.array(xx[0]) # 其他翼型的横坐标是一样的
  r = len(yy) # 总共有多少个翼型
  # preallocate parameter array
  # last column is for the error
  opt_params = np.zeros([r,10]) 

  params_0 = np.zeros(9) + 0.05

  t1 = time()
  for i in range(r):
    print(str(i+1)+'/'+str(r))
    bnds =((1e-6,10),(1e-6,10),(-10,10),(-100,100),(-75,75),(1e-6,10),(-10,10),(-100,100),(-75,75))
    res = minimize(penalty, params_0,args=(i),tol=1e-8, options={'maxiter': 5000}, bounds=bnds)
    if res.success == False:
        print("Optimization Failed")
    opt_params[i,0:9] = np.array(res.x)
    opt_params[i,-1] = res.fun

  t2 = time()

  elapsed = t2 - t1
  print(elapsed)

  # plot the best case and the worst case
  ind_best = np.argmin(opt_params[:,-1])
  ind_worst = np.argmax(opt_params[:,-1])
  y_params_best = opt_params[ind_best,:-1]
  y_params_best = params_to_coord(y_params_best)
  y_params_worst = opt_params[ind_worst,:-1]
  y_params_worst = params_to_coord(y_params_worst)
  y_data_best = y_coord_data[ind_best,:]
  y_data_worst = y_coord_data[ind_worst,:]

  fig,ax = plt.subplots(1, 2, figsize=(10, 5))
  ax[0].plot(x_coord_data,y_params_best, label='parameterized airfoil')
  ax[0].plot(x_coord_data,y_data_best, label='actual airfoil')
  ax[0].legend()
  ax[0].set_title('Best Parameterization')
#   ax[0].set_aspect(5) # hard coded
  ax[1].plot(x_coord_data,y_params_worst, label='parameterized airfoil')
  ax[1].plot(x_coord_data,y_data_worst, label='actual airfoil')
  ax[1].legend()
  ax[1].set_title('Worst Parameterization')
#   ax[1].set_aspect(2) # hard coded
  plt.tight_layout()
  plt.savefig('best_worst.png',dpi=300)
  plt.show()

#   # save params file, which is used by the airfoil generator
#   parsec_params_path = 'data/airfoil/parsec_params.txt'
#   # 需要再之前加入文件file_name
#   with open(parsec_params_path,'w') as f:
#     for i,path in enumerate(allData):
#       f.write(path)
#       f.write(',')
#       f.write(','.join(map(str,opt_params[i,:])))
#       f.write('\n')
       