import torch
from torch.utils.data import Dataset,DataLoader
import os
import torchvision.transforms as transforms
import numpy as np

class EditingAirFoilDataset(Dataset):
    """Dataset for shape datasets(coco & 机翼)"""
    def __init__(self,split = 'train',
                 datapath = './data/airfoil/picked_uiuc',
                 ):
        self.split = split
        self.datapath = datapath
        split = 'editing_' + split 
        with open('data/airfoil/%s.txt' % split) as f:
              txt_list = [(os.path.join(datapath,line.rstrip().strip('\n').split(',')[0] + '.dat',) ,
                            os.path.join(datapath,line.rstrip().strip('\n').split(',')[1] + '.dat',))
                          for line in f.readlines()]
        self.txt_list = txt_list
        self.params = {}
        with open('data/airfoil/parsec_params.txt') as f:
            for line in f.readlines():
                name_params = line.rstrip().strip('\n').split(',')
                # 取出路径的最后一个文件名作为key
                name = os.path.basename(name_params[0].split('.')[0]).strip()# name_params[0].split('/')[-1].split('.')[0]
                # print('name ', name, ' value ', list(map(float,name_params[1:])))
                self.params[name] = list(map(float,name_params[1:]))
        print("dataset init done. len: ", len(txt_list))
        
    def __getitem__(self, index):
        """Get current batch for input index"""
        txt_path1, txt_path2 = self.txt_list[index]
        key1 = os.path.basename(txt_path1).split('.')[0].strip()# txt_path1.split('/')[-1].split('.')[0]
        key2 = os.path.basename(txt_path2).split('.')[0].strip()# txt_path2.split('/')[-1].split('.')[0]
        params1 = self.params[key1]
        params2 = self.params[key2]
        # ---
        data = []
        with open(txt_path1) as file:
            # 逐行读取文件内容
            for line in file:
                # 移除行尾的换行符，并将每行拆分为两个数值
                values = line.strip().split()
                # 将数值转换为浮点数，并添加到数据列表中
                data.append([float(values[0]), float(values[1])])
        if len(data) == 201:
            data = data[:200]
        elif len(data) > 201:
            assert len(data) < 201, f'data is not 200 ! {txt_path1}'
        # data = self.pc_norm(data)
        data = torch.FloatTensor(data)
        input = data[::10] # 20个点
        # ---
        data2 = []
        with open(txt_path2) as file:
            # 逐行读取文件内容
            for line in file:
                # 移除行尾的换行符，并将每行拆分为两个数值
                values = line.strip().split()
                # 将数值转换为浮点数，并添加到数据列表中
                data2.append([float(values[0]), float(values[1])])
        if len(data2) == 201:
            data2 = data2[:200]
        elif len(data2) > 201:
            assert len(data2) < 201, f'data is not 200 ! {txt_path2}'
        data2 = torch.FloatTensor(data2)
        input2 = data2[::10] # 20个点
        # ---
        params1 = torch.FloatTensor(params1)
        params2 = torch.FloatTensor(params2)
        return {'origin_input':input, 'editing_input':input2, 'params':params1, 'gt':params2, 'origin_data':data,'origin_trans':data2,'tx1':txt_path1,'tx2':txt_path2}
    
    def __len__(self):
        return len(self.txt_list)
    
    def pc_norm(self,pc):
        """ pc: NxC, return NxC """
        centroid = np.mean(pc, axis=0)
        pc = pc - centroid
        m = np.max(np.sqrt(np.sum(pc**2, axis=1)))
        return pc / m


class EditingDataset(Dataset):
    """Dataset for shape datasets(coco & 机翼)"""
    def __init__(self,split = 'train',
                 datapath = 'data/airfoil/supercritical_airfoil',
                 ):
        self.split = split
        self.datapath = datapath
        
        with open('data/airfoil/%s.txt' % split) as f:
              txt_list = [os.path.join(datapath,line.rstrip().strip('\n') + '.dat',) 
                          for line in f.readlines()]
        self.txt_list = txt_list
        self.params = {}
        # params = []
        with open('data/airfoil/parsec_params_direct.txt') as f:
            for line in f.readlines():
                name_params = line.rstrip().strip('\n').split(',')
                # 取出路径的最后一个文件名作为key
                name = name_params[0].split('/')[-1].split('.')[0]
                self.params[name] = list(map(float,name_params[1:]))

    def get_data(self, txt_path):
        data = []
        with open(txt_path) as file:
            # 逐行读取文件内容
            for line in file:
                # 移除行尾的换行符，并将每行拆分为两个数值
                values = line.strip().split()
                # 将数值转换为浮点数，并添加到数据列表中
                data.append([float(values[0]), float(values[1])])
        data = torch.FloatTensor(data)
        return data

    def __getitem__(self, index):
        """Get current batch for input index"""
        index2 = np.random.randint(len(self.txt_list))
        txt_path1 = self.txt_list[index]
        txt_path2 = self.txt_list[index2]
        key1 = txt_path1.split('/')[-1].split('.')[0]
        key2 = txt_path2.split('/')[-1].split('.')[0]
        params1 = torch.FloatTensor(self.params[key1])
        params2 = torch.FloatTensor(self.params[key2])
        source = self.get_data(txt_path1)
        target = self.get_data(txt_path2)
        source_keypoint = source[::10] # 26个点
        target_keypoint = target[::10] # 26个点
        return {'source_keypoint':source_keypoint,'source_point':source,'source_param':params1,
                'target_keypoint':target_keypoint,'target_point':target,'target_param':params2}
    
    def __len__(self):
        return len(self.txt_list)


if __name__=='__main__':
    dataset = EditingDataset()
    dataloader = DataLoader(dataset,batch_size=2,shuffle=True,num_workers=0)
    for i,data in enumerate(dataloader):
        print(data)
        break
    print(len(dataloader))
        
        
        
# srun -p Gvlab-S1-32 --quotatype=spot -n2 --gres=gpu:2 --ntasks-per-node 2 python demos/multi_turn_mm.py --n_gpus=2 --tokenizer_path=../
# llama2/tokenizer.model --llama_type=llama_ens --pretrained_path ../ckpt_path --llama_config ./13bconfig.json