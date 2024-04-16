import os
import torch
import torch.nn as nn
from tqdm import tqdm 
from torch.utils.data import DataLoader
import torch.optim as optim 
import torch.nn.functional as F
import torch.optim.lr_scheduler as lr_scheduler
from models import CVAE_Y
from dataload import AirFoilMixParsec, Fit_airfoil
import math 
from utils import vis_airfoil2
import random
import argparse
import numpy as np

attention_type = 'cross'
logs_name = f'cvae_y_{attention_type}_attention'
os.makedirs(f'logs/{logs_name}',exist_ok=True)
os.makedirs(f'weights/{logs_name}',exist_ok=True)

os.environ['CUDA_LAUNCH_BLOCKING'] = '1'

def parse_option():
    """Parse cmd arguments."""
    parser = argparse.ArgumentParser()
    # Data
    parser.add_argument('--batch_size', type=int, default=512,
                        help='Batch Size during training')
    parser.add_argument('--latent_size', type=int, default=20,
                        help='Batch Size during training')
    parser.add_argument('--num_workers',type=int,default=8)
    # Training
    parser.add_argument('--start_epoch', type=int, default=1)
    parser.add_argument('--max_epoch', type=int, default=1101)
    parser.add_argument('--weight_decay', type=float, default=0.0005)
    parser.add_argument("--lr", default=1e-3, type=float)
    parser.add_argument('--lrf', type=float, default=0.01)
    parser.add_argument('--lr-scheduler', type=str, default='step',
                          choices=["step", "cosine"])
    parser.add_argument('--lr_decay_epochs', type=int, default=[50, 75],
                        nargs='+', help='when to decay lr, can be a list')
    parser.add_argument('--lr_decay_rate', type=float, default=0.1,
                        help='for step scheduler. decay rate for lr')
    parser.add_argument('--warmup-epoch', type=int, default=-1)
    parser.add_argument('--warmup-multiplier', type=int, default=100)

    # io
    parser.add_argument('--checkpoint_path', default='weights/cvae_y_cross_attention/ckpt_epoch_1000.pth',help='Model checkpoint path') # ./eval_result/logs_p/ckpt_epoch_last.pth
    parser.add_argument('--log_dir', default=f'weights/{logs_name}',
                        help='Dump dir to save model checkpoint')
    parser.add_argument('--val_freq', type=int, default=100)  # epoch-wise
    parser.add_argument('--save_freq', type=int, default=100)  # epoch-wise
    

    # 评测指标相关
    parser.add_argument('--distance_threshold', type=float, default=0.01) # xy点的距离小于该值，被认为是预测正确的点
    parser.add_argument('--threshold_ratio', type=float, default=0.75) # 200个点中，预测正确的点超过一定的比例，被认为是预测正确的样本
   
    args, _ = parser.parse_known_args()


    return args

# BRIEF load checkpoint.
def load_checkpoint(args, model, optimizer, scheduler):
    """Load from checkpoint."""
    print("=> loading checkpoint '{}'".format(args.checkpoint_path))

    checkpoint = torch.load(args.checkpoint_path, map_location='cpu')
    try:
        args.start_epoch = int(checkpoint['epoch'])
    except Exception:
        args.start_epoch = 1
    model.load_state_dict(checkpoint['model'], strict=True)
    optimizer.load_state_dict(checkpoint['optimizer'])
    scheduler.load_state_dict(checkpoint['scheduler'])

    print("=> loaded successfully '{}' (epoch {})".format(
        args.checkpoint_path, checkpoint['epoch']
    ))

    del checkpoint
    torch.cuda.empty_cache()

# Reconstruction + KL divergence losses summed over all elements and batch
def loss_function(recon_x, x, mu, logvar):
    recons = nn.MSELoss()(recon_x, x)
    # see Appendix B from VAE paper:
    # Kingma and Welling. Auto-Encoding Variational Bayes. ICLR, 2014
    # https://arxiv.org/abs/1312.6114
    # 0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
    KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    # device = x.device
    # f = torch.FloatTensor(Fit_airfoil(x[0].detach().cpu().numpy()).parsec_features).to(device)
    # f_gt = torch.FloatTensor(Fit_airfoil(recon_x.reshape(-1,257,2)[0].detach().cpu().numpy()).parsec_features).to(device)
    # 用离散方法计算物理量的误差，嵌入到计算图，梯度反传


    # 根据 物理量的误差调整x，将x project x'
    # reconx = recon_x.reshape(-1,257,2)
    # recons_x_new = project(recon_x,f,f_gt) # project function TODO
    # recons_loss_new = nn.MSELoss()(recons_x_new, x)
    return recons + KLD

# BRIEF save model.
def save_checkpoint(args, epoch, model, optimizer, scheduler, save_cur=False):
    """Save checkpoint if requested."""
    if epoch % args.save_freq == 0:
        state = {
            'config': args,
            'save_path': '',
            'model': model.state_dict(),
            'optimizer': optimizer.state_dict(),
            'scheduler': scheduler.state_dict(),
            'epoch': epoch
        }
        os.makedirs(args.log_dir, exist_ok=True)
        spath = os.path.join(args.log_dir, f'ckpt_epoch_{epoch}.pth')
        state['save_path'] = spath
        torch.save(state, spath)
        print("Saved in {}".format(spath))
    else:
        print("not saving checkpoint")


class Trainer:

    def get_datasets(self):
        """获得训练、验证 数据集"""
        train_dataset = AirFoilMixParsec(split='train',dataset_names=['r05','r06','supercritical_airfoil','interpolated_uiuc'])
        val_dataset = AirFoilMixParsec(split='val',dataset_names=['r05','r06','supercritical_airfoil','interpolated_uiuc'])
        return train_dataset, val_dataset
    
    def get_loaders(self,args):
        """获得训练、验证 dataloader"""
        print("get_loaders func begin, loading......")
        train_dataset, val_dataset = self.get_datasets()
        train_loader = DataLoader(train_dataset,
                                  shuffle=True,
                                  batch_size=args.batch_size,
                                  num_workers=args.num_workers)
        val_loader = DataLoader(val_dataset,
                                  shuffle=False,
                                  batch_size=args.batch_size,
                                  num_workers=args.num_workers)
        return train_loader,val_loader

    @staticmethod
    def get_model(args):
        model = CVAE_Y(attention_type=attention_type)
        return model

    @staticmethod
    def get_optimizer(args,model):
        params = [p for p in model.parameters() if p.requires_grad]
        optimizer = optim.AdamW(params,
                              lr = args.lr,
                              weight_decay=args.weight_decay)
        return optimizer
    


    def train_one_epoch(self,model,optimizer,dataloader,device,epoch):
        """训练一个epoch"""
        model.train()  # set model to training mode
        total_loss = 0
        total_pred = 0
        for _,data in enumerate(tqdm(dataloader)):
            keypoint = data['keypoint'][:,:,1:2] # [b,26,1]
            gt = data['gt'][:,:,1:2] # [b,257,1]
            physics = data['params'] # [b,11]
            physics = physics.unsqueeze(-1) # [b,11,1]
            condition = torch.cat([physics,keypoint],dim=1)
            condition = condition.to(device)
            gt = gt.to(device)
            optimizer.zero_grad()

            recon_batch, mu, logvar = model(gt,condition) # [b,257],[b,37] -> [b,257]

            loss = loss_function(recon_batch, gt, mu, logvar)

            total_loss += loss.item()
            total_pred += keypoint.shape[0]
            # 反向传播
            loss.backward()
            # 参数更新
            optimizer.step()    
        total_loss = total_loss / total_pred        
        # 打印loss
        print('====> Epoch: {} Average loss: {:.8f}'.format(
          epoch, total_loss))


    @torch.no_grad()
    def evaluate_one_epoch(self, model, dataloader,device, epoch, args):
        """验证一个epoch"""
        model.eval()

        total_parsec_loss = [0]*11
        
        correct_pred = 0  # 预测正确的样本数量
        total_pred = 0  # 总共的样本数量
        total_loss = recons_loss = 0.0

        test_loader = tqdm(dataloader)
        for _,data in enumerate(test_loader):
            
            keypoint = data['keypoint'][:,:,1:2] # [b,26,1]
            x = data['gt'][0,:,0] # [257]
            gt = data['gt'][:,:,1:2] # [b,257,1]
            physics = data['params'] # [b,11]
            physics = physics.unsqueeze(-1) # [b,11,1]
            condition = torch.cat([physics,keypoint],dim=1)
            condition = condition.to(device)
            gt = gt.to(device)

            recon_batch, mu, logvar = model(gt,condition) # [b,257],[b,37] -> [b,257]


            total_pred += keypoint.shape[0]
            loss = loss_function(recon_batch, gt, mu, logvar)
            recons_loss += nn.MSELoss()(recon_batch, gt)
            total_loss += loss.item()
            # 判断样本是否预测正确
            distances = torch.norm(gt - recon_batch,dim=-1) #(B,257)

            # 点的直线距离小于t，说明预测值和真实值比较接近，认为该预测值预测正确
            t = args.distance_threshold
            # 257个点中，预测正确的点的比例超过ratio，认为该形状预测正确
            ratio = args.threshold_ratio
            count = (distances < t).sum(1) #(B) 一个样本中预测坐标和真实坐标距离小于t的点的个数
            correct_count = (count >= ratio*257).sum().item() # batch_size数量的样本中，正确预测样本的个数
            correct_pred += correct_count

            # 统计一下物理量之间的误差
            for idx in range(recon_batch.shape[0]):
                # 给他们拼接同一个x坐标
                source = recon_batch[idx,:,0].detach().cpu().numpy() # [257]
                target = gt[idx,:,0].detach().cpu().numpy() # [257]
                source = np.stack([x,source],axis=1)
                target = np.stack([x,target],axis=1)
                source_parsec = Fit_airfoil(source).parsec_features
                target_parsec = Fit_airfoil(target).parsec_features
                for i in range(11):
                    total_parsec_loss[i] += abs(source_parsec[i]-target_parsec[i])/(abs(target_parsec[i])+1e-9)
                # if idx < 5 and epoch % 100 == 0:
                #   vis_airfoil2(source,target,epoch+idx,dir_name='logs/cvae_y',sample_type='cvae')

        accuracy = correct_pred / total_pred
        avg_loss = total_loss / total_pred
        recons_loss = recons_loss / total_pred
        avg_parsec_loss = [x/total_pred for x in total_parsec_loss]
        
        print(f"eval——epoch: {epoch}, accuracy: {accuracy}, avg_loss: {avg_loss}, recons_loss: {recons_loss}")
        print(f"eval——epoch: {epoch}, avg_parsec_loss: {avg_parsec_loss}")
        # 保存评测结果
        with open(f'logs/{logs_name}/eval_result.txt','a') as f:
            f.write(f"eval——epoch: {epoch}, accuracy: {accuracy}, avg_loss: {avg_loss}\n")
            f.write(f"eval——epoch: {epoch}, avg_parsec_loss: {avg_parsec_loss}\n")
    

    @torch.no_grad()
    def infer(self, model, dataloader,device, epoch, args):
        """验证一个epoch"""
        model.eval()

        total_parsec_loss = [0]*11
        
        correct_pred = 0  # 预测正确的样本数量
        total_pred = 0  # 总共的样本数量
        total_loss = 0.0

        test_loader = tqdm(dataloader)
        for _,data in enumerate(test_loader):
            
            keypoint = data['keypoint'][:,:,1:2] # [b,26,1]
            x = data['gt'][0,:,0] # [257]
            gt = data['gt'][:,:,1:2] # [b,257,1]
            physics = data['params'] # [b,11]
            physics = physics.unsqueeze(-1) # [b,11,1]
            condition = torch.cat([physics,keypoint],dim=1)
            condition = condition.to(device)
            gt = gt.to(device)

            z = torch.randn((gt.shape[0],args.latent_size)).to(device)
            recon_batch = model.decode(z,condition).reshape(-1,x.shape[0],1) # [b,20],[b,37,1] -> [b,257,1]

            total_pred += keypoint.shape[0]

            loss = nn.MSELoss()(recon_batch, gt)
            total_loss += loss.item()
            # 判断样本是否预测正确
            distances = torch.norm(gt - recon_batch,dim=-1) #(B,257)

            # 点的直线距离小于t，说明预测值和真实值比较接近，认为该预测值预测正确
            t = args.distance_threshold
            # 257个点中，预测正确的点的比例超过ratio，认为该形状预测正确
            ratio = args.threshold_ratio
            count = (distances < t).sum(dim=1) #(B) 一个样本中预测坐标和真实坐标距离小于t的点的个数
            correct_count = (count >= ratio*257).sum().item() # batch_size数量的样本中，正确预测样本的个数
            correct_pred += correct_count

            # 统计一下物理量之间的误差
            for idx in range(recon_batch.shape[0]):
                # 给他们拼接同一个x坐标
                source = recon_batch[idx][:,0].detach().cpu().numpy() # [257]
                target = gt[idx][:,0].detach().cpu().numpy() # [257]

                # 需要check 一下为啥x直接就是numpy格式
                source = np.stack([x,source],axis=1)
                target = np.stack([x,target],axis=1)
                source_parsec = Fit_airfoil(source).parsec_features
                target_parsec = Fit_airfoil(target).parsec_features
                for i in range(11):
                    total_parsec_loss[i] += abs(source_parsec[i]-target_parsec[i])
                if idx < 5:
                  vis_airfoil2(source,target,epoch+idx,dir_name=f'logs/{logs_name}',sample_type='cvae')

        accuracy = correct_pred / total_pred
        avg_loss = total_loss / total_pred
        avg_parsec_loss = [x/total_pred for x in total_parsec_loss]
        
        print(f"infer——epoch: {epoch}, accuracy: {accuracy}, avg_loss: {avg_loss}")
        print(f"infer——epoch: {epoch}, avg_parsec_loss: {avg_parsec_loss}")
        # 保存评测结果
        with open(f'logs/{logs_name}/infer_result.txt','a') as f:
            f.write(f"infer——epoch: {epoch}, accuracy: {accuracy}, avg_loss: {avg_loss}\n")
            f.write(f"infer——epoch: {epoch}, avg_parsec_loss: {avg_parsec_loss}\n")

    def main(self,args):
        """Run main training/evaluation pipeline."""

        # 单卡训练
        model = self.get_model(args)
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model.to(device)

        train_loader, val_loader = self.get_loaders(args) 
        optimizer = self.get_optimizer(args,model)
        # Scheduler https://arxiv.org/pdf/1812.01187.pdf
        lf = lambda x: ((1 + math.cos(x * math.pi / args.max_epoch)) / 2) * (1 - args.lrf) + args.lrf  # cosine
        scheduler = lr_scheduler.LambdaLR(optimizer, lr_lambda=lf)
      
 
        # Check for a checkpoint
        if len(args.checkpoint_path)>0:
            assert os.path.isfile(args.checkpoint_path)
            load_checkpoint(args, model, optimizer, scheduler)
            
        for epoch in range(args.start_epoch,args.max_epoch+1):
            # # train
            self.train_one_epoch(model=model,
                                 optimizer=optimizer,
                                 dataloader=train_loader,
                                 device=device,
                                 epoch=epoch
                                 )
            scheduler.step()
            # save model and validate
            # args.val_freq = 1
            if epoch % args.val_freq == 0:
                save_checkpoint(args, epoch, model, optimizer, scheduler)
                print("Validation begin.......")
                self.evaluate_one_epoch(
                    model=model,
                    dataloader=val_loader,
                    device=device, 
                    epoch=epoch, 
                    args=args)
                self.infer(
                    model=model,
                    dataloader=val_loader,
                    device=device, 
                    epoch=epoch, 
                    args=args)
           
          
                

if __name__ == '__main__':
    opt = parse_option()
    # Fix random seed for reproducibility
    seed = 42
    torch.manual_seed(seed)
    random.seed(seed)


    torch.backends.cudnn.enabled = True
    # 启用cudnn的自动优化模式
    torch.backends.cudnn.benchmark = True
    # 设置cudnn的确定性模式，pytorch确保每次运行结果的输出都保持一致
    torch.backends.cudnn.deterministic = True

    trainer = Trainer()
    trainer.main(opt)