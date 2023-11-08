from __future__ import division
from math import sqrt, tan, pi
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from numpy.lib.shape_base import expand_dims
import pandas as pd
from scipy.optimize import minimize
from time import time
import os
from tqdm import tqdm

# INPUT.csv must be placed in the same directory as the script

# Code copied and modified from https://github.com/dqsis/parsec-airfoils
def pcoef(
        xte,yte,rle,
        x_cre,y_cre,d2ydx2_cre,th_cre,
        surface):
    """evaluate the PARSEC coefficients"""

    # Initialize coefficients
    coef = np.zeros(6)

    # 1st coefficient depends on surface (pressure or suction)
    if surface.startswith('p'):
        coef[0] = -sqrt(2*rle)
    else:
        coef[0] = sqrt(2*rle)
 
    # Form system of equations
    A = np.array([
                 [xte**1.5, xte**2.5, xte**3.5, xte**4.5, xte**5.5],
                 [x_cre**1.5, x_cre**2.5, x_cre**3.5, x_cre**4.5, 
                  x_cre**5.5],
                 [1.5*sqrt(xte), 2.5*xte**1.5, 3.5*xte**2.5, 
                  4.5*xte**3.5, 5.5*xte**4.5],
                 [1.5*sqrt(x_cre), 2.5*x_cre**1.5, 3.5*x_cre**2.5, 
                  4.5*x_cre**3.5, 5.5*x_cre**4.5],
                 [0.75*(1/sqrt(x_cre)), 3.75*sqrt(x_cre), 8.75*x_cre**1.5, 
                  15.75*x_cre**2.5, 24.75*x_cre**3.5]
                 ]) 

    B = np.array([
                 [yte - coef[0]*sqrt(xte)],
                 [y_cre - coef[0]*sqrt(x_cre)],
                 [tan(th_cre*pi/180) - 0.5*coef[0]*(1/sqrt(xte))],
                 [-0.5*coef[0]*(1/sqrt(x_cre))],
                 [d2ydx2_cre + 0.25*coef[0]*x_cre**(-1.5)]
                 ])
    
    # Solve system of linear equations
    try:
        X = np.linalg.solve(A,B)
    except:
        X = np.linalg.solve(A+(1e-12*np.eye(5)),B)


    # Gather all coefficients
    coef[1:6] = X[0:5,0]

    # Return coefficients
    return coef


def ppoints(cf_pre, cf_suc, xte=1.0):
    '''
    Takes PARSEC coefficients, number of points, and returns list of
    [x,y] coordinates starting at trailing edge pressure side.
    Assumes trailing edge x position is 1.0 if not specified.
    Returns 121 points if 'npts' keyword argument not specified.
    '''
    # Using cosine spacing to concentrate points near TE and LE,
    # see http://airfoiltools.com/airfoil/naca4digit
    # modfied xpts 直接由数据读入
    ind = np.where(np.array(x_coord_data)==0)[0][0]
    xpts = x_coord_data[:ind+1] # 从X_te:1 , 到 X_le:0
    # Powers to raise coefficients to
    pwrs = (1/2, 3/2, 5/2, 7/2, 9/2, 11/2)
    # Make [[1,1,1,1],[2,2,2,2],...] style array
    xptsgrid = np.meshgrid(np.arange(len(pwrs)), xpts)[1]
    # Evaluate points with concise matrix calculations. One x-coordinate is
    # evaluated for every row in xptsgrid
    evalpts = lambda cf: np.sum(cf*xptsgrid**pwrs, axis=1)
    # Move into proper order: start at TE, over bottom, then top
    # Avoid leading edge pt (0,0) being included twice by slicing [1:]

    ycoords = np.append(evalpts(cf_suc), evalpts(cf_pre)[1:-1][::-1]) # 
    xcoords = np.append(xpts, xpts[1:-1][::-1]) # 先是上表面，后下表面

    # Return 2D list of coordinates [[x,y],[x,y],...] by transposing .T
    # Return lower surface then upper surface
    return np.array((xcoords, ycoords)).T


def params_to_coord(params):
    '''wrapper function to convert parsec parameters
        to coordinates and get only y coordinates'''
    # TE & LE of airfoil (normalized, chord = 1)
    xle = 0.0
    yle = 0.0
    xte = 1.0
    yte = 0.0

    rle = params[0] # leading edge radius

    # Pressure (lower) surface parameters 
    x_pre = params[1] # pressure crest location x-coordinate
    y_pre = params[2] # pressure crest location y-coordinate
    d2ydx2_pre = params[3] # curvatures at the pressuresurface crest locations
    th_pre = params[4] # angle of the pressure surface at the trailing edge

    # Suction (upper) surface parameters
    x_suc = params[5] # suction crest location x-coordinate
    y_suc = params[6] # suction crest location y-coordinate
    d2ydx2_suc = params[7] # curvatures at suction surface crest locations
    th_suc = params[8] # angle of the suction surface at the trailing edge
    cf_pre = pcoef(xte,yte,rle,x_pre,y_pre,d2ydx2_pre,th_pre,'pre') # 下表面
    cf_suc = pcoef(xte,yte,rle,x_suc,y_suc,d2ydx2_suc,th_suc,'suc') # 上表面
    coord = ppoints(cf_pre, cf_suc,xte=1.0) # 得到的是200个点的坐标, 按照逆时针顺序来的
    y_coord = coord[:,1] # 只取y坐标
    return y_coord


def calMidLine(y_data):
    mid_data = []
    for i in range(101):
        if i==0 or i==100:
            mid_data.append(0)
        else:
            mid_data.append((y_data[i]+y_data[-i])/2)
    return np.array(mid_data)

def penalty(opt_params,opt_ind):
    '''compute penalty, which is sum of square root of difference between the data and
        coordinates generated from the parameters'''
    y_data = y_coord_data[opt_ind,:] # 选取第opt_ind个翼型
    y_params = params_to_coord(opt_params)
    
    pen_elem = np.square(y_data-y_params)

    # ## 中轴线
    mid_data = calMidLine(y_data)
    mid_params = calMidLine(y_params)
    pen_elem1 = np.square(mid_data-mid_params)

    pen = np.sum(pen_elem)
    pen += np.sum(pen_elem1)
    return pen


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
       