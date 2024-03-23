import os


root_path = 'data/airfoil/supercritical_airfoil'
allData = []
## 使用os.walk()函数遍历文件夹
for root, dirs, files in os.walk(root_path):
    for file in files:
        file_path = os.path.join(root, file)
        # Do something with the file_path
        allData.append(file_path)

print(len(allData)) # 读取了所有的飞机数据
data = allData[:50] # 采样10个飞机数据

## 使用sub子图，2*5个子图

import matplotlib.pyplot as plt

fig, axs = plt.subplots(10, 5, figsize=(12, 6))

for i, file_path in enumerate(data):
    ax = axs[i // 5, i % 5]  # Get the appropriate subplot
    # Load and plot the data
    # Assuming the data is in a text file with one point per line
    with open(file_path, 'r') as f:
        lines = f.readlines()
        x = []
        y = []
        for line in lines:
            point = line.strip().split()
            x.append(float(point[0]))
            y.append(float(point[1]))
        ax.plot(x, y) # 画出飞机的轮廓
        ax.set_aspect('equal')
        ax.axis('off')

fig.suptitle('Airfoil Outlines')
plt.tight_layout()  # Adjust the spacing between subplots
plt.savefig('data/Aircraft.png')
plt.show()

