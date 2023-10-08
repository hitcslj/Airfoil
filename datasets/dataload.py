import torch
from torch.utils.data import Dataset,DataLoader
import os
import torchvision.transforms as transforms
import numpy as np

class AirFoilDataset(Dataset):
    """Dataset for shape datasets(coco & 机翼)"""
    def __init__(self,split = 'train',
                 datapath = './data/airfoil/picked_uiuc',
                 ):
        self.split = split
        self.datapath = datapath
        
        with open('data/airfoil/%s.txt' % split) as f:
              txt_list = [os.path.join(datapath,line.rstrip().strip('\n') + '.dat',) 
                          for line in f.readlines()]
        self.txt_list = txt_list
        self.params = {}
        with open('data/airfoil/parsec_params.txt') as f:
            for line in f.readlines():
                name_params = line.rstrip().strip('\n').split(',')
                # 取出路径的最后一个文件名作为key
                name = name_params[0].split('/')[-1].split('.')[0]
                self.params[name] = list(map(float,name_params[1:]))
        
    def __getitem__(self, index):
        """Get current batch for input index"""
        txt_path = self.txt_list[index]
        key = txt_path.split('/')[-1].split('.')[0]
        params = self.params[key]
        data = []
        with open(txt_path) as file:
            # 逐行读取文件内容
            for line in file:
                # 移除行尾的换行符，并将每行拆分为两个数值
                values = line.strip().split()
                # 将数值转换为浮点数，并添加到数据列表中
                data.append([float(values[0]), float(values[1])])
        if len(data) == 201:
            data = data[:200]
        elif len(data) > 201:
            assert len(data) < 201, f'data is not 200 ! {txt_path}'
        # data = self.pc_norm(data)
        data = torch.FloatTensor(data)
        # input = data[::10]
        # params = torch.FloatTensor(params)
        # return {'input':input,'output':data,'params':params}
        return data
    
    def __len__(self):
        return len(self.txt_list)
    
    def pc_norm(self,pc):
        """ pc: NxC, return NxC """
        centroid = np.mean(pc, axis=0)
        pc = pc - centroid
        m = np.max(np.sqrt(np.sum(pc**2, axis=1)))
        return pc / m


class AirFoilDataset2(Dataset):
    """Dataset for shape datasets(coco & 机翼)"""
    def __init__(self,split = 'train',
                 datapath = './data/airfoil/picked_uiuc',
                 ):
        self.split = split
        self.datapath = datapath
        
        with open('data/airfoil/%s.txt' % split) as f:
              txt_list = [os.path.join(datapath,line.rstrip().strip('\n') + '.dat',) 
                          for line in f.readlines()]
        self.txt_list = txt_list
        self.params = {}
        with open('data/airfoil/parsec_params.txt') as f:
            for line in f.readlines():
                name_params = line.rstrip().strip('\n').split(',')
                # 取出路径的最后一个文件名作为key
                name = name_params[0].split('/')[-1].split('.')[0]
                self.params[name] = list(map(float,name_params[1:]))
        
    def __getitem__(self, index):
        """Get current batch for input index"""
        txt_path = self.txt_list[index]
        key = txt_path.split('/')[-1].split('.')[0]
        params = self.params[key]
        data = []
        with open(txt_path) as file:
            # 逐行读取文件内容
            for line in file:
                # 移除行尾的换行符，并将每行拆分为两个数值
                values = line.strip().split()
                # 将数值转换为浮点数，并添加到数据列表中
                data.append([float(values[0]), float(values[1])])
        if len(data) == 201:
            data = data[:200]
        elif len(data) > 201:
            assert len(data) < 201, f'data is not 200 ! {txt_path}'
        # data = self.pc_norm(data)
        data = torch.FloatTensor(data)
        input = data[::10] # 20个点
        params = torch.FloatTensor(params)
        return {'input':input,'output':data,'params':params}
    
    def __len__(self):
        return len(self.txt_list)
    
    def pc_norm(self,pc):
        """ pc: NxC, return NxC """
        centroid = np.mean(pc, axis=0)
        pc = pc - centroid
        m = np.max(np.sqrt(np.sum(pc**2, axis=1)))
        return pc / m


if __name__=='__main__':
    dataset = AirFoilDataset2()
    dataloader = DataLoader(dataset,batch_size=1,shuffle=True,num_workers=4)
    for i,data in enumerate(dataloader):
        print(data)
        break