import os
import shutil
import numpy as np

source_path = 'data/airfoil/supercritical_airfoil'
 

def read_file(file_path):
    data = []
    with open(file_path) as file:
        for i,line in enumerate(file):
            if i==0:continue
            values = line.strip().split()
            data.append([float(values[0]), float(values[2])])
    return np.array(data)

for root, dirs, files in os.walk(source_path):
  for file in files:
      file_path = os.path.join(root, file)
      data = read_file(file_path)
      # 将data重写写入到file_path中
      with open(file_path, 'w') as f:
        for row in data:
          f.write(f'{row[0]} {row[1]}\n')

      
      