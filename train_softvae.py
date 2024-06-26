import os
import torch
import torch.nn as nn
from tqdm import tqdm 
from torch.utils.data import DataLoader
import torch.optim as optim 
import torch.nn.functional as F
import torch.optim.lr_scheduler as lr_scheduler
from models import Soft_VAE
from dataload import AirFoilMixParsec, Fit_airfoil
import math 
from utils import vis_airfoil2
import random
import argparse
import numpy as np
from utils import cal_diversity_score,calculate_smoothness


os.environ['CUDA_LAUNCH_BLOCKING'] = '1'

def parse_option():
    """Parse cmd arguments."""
    parser = argparse.ArgumentParser()
    # Data
    parser.add_argument('--batch_size', type=int, default=1024,
                        help='Batch Size during training')
    parser.add_argument('--latent_size', type=int, default=10,
                        help='Batch Size during training')
    parser.add_argument('--num_workers',type=int,default=4)
    # Training
    parser.add_argument('--beta', default=1, type=float, help='beta parameter for KL-term in original beta-VAE')
    parser.add_argument('--start_epoch', type=int, default=1)
    parser.add_argument('--max_epoch', type=int, default=1001)
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
    parser.add_argument('--checkpoint_path', default='',help='Model checkpoint path') # logs/soft_vae/ckpt_epoch_1000.pth
    parser.add_argument('--log_dir', default=f'logs/soft_vae',
                        help='Dump dir to save model checkpoint & experiment log')
    parser.add_argument('--val_freq', type=int, default=1000)  # epoch-wise
    parser.add_argument('--save_freq', type=int, default=1000)  # epoch-wise
    

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
    recons = nn.MSELoss(reduction='mean')(recon_x.squeeze(-1), x.squeeze(-1))
    # see Appendix B from VAE paper:
    # Kingma and Welling. Auto-Encoding Variational Bayes. ICLR, 2014
    # https://arxiv.org/abs/1312.6114
    # 0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
    KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    return recons, KLD

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
        train_dataset = AirFoilMixParsec(split='train',dataset_names=['cst_gen','supercritical_airfoil','interpolated_uiuc'])  
        test_dataset = AirFoilMixParsec(split='test',dataset_names=['cst_gen','supercritical_airfoil','interpolated_uiuc']) 
        return train_dataset, test_dataset
    
    def get_loaders(self,args):
        """获得训练、验证 dataloader"""
        print("get_loaders func begin, loading......")
        train_dataset, test_dataset = self.get_datasets()
        train_loader = DataLoader(train_dataset,
                                  shuffle=True,
                                  batch_size=args.batch_size,
                                  num_workers=args.num_workers)
        test_loader = DataLoader(test_dataset,
                                shuffle=False,
                                batch_size=args.batch_size,
                                num_workers=args.num_workers)
        return train_loader,test_loader

    @staticmethod
    def get_model(args):
        model = Soft_VAE()
        return model

    @staticmethod
    def get_optimizer(args,model):
        params = [p for p in model.parameters() if p.requires_grad]
        optimizer = optim.AdamW(params,
                              lr = args.lr,
                              weight_decay=args.weight_decay)
        return optimizer
    


    def train_one_epoch(self,args, model,optimizer,dataloader,device,epoch):
        """训练一个epoch"""
        model.train()  # set model to training mode
        total_recons_loss = total_kl_loss = 0
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

            recon_batch, mu, logvar = model(gt,condition) # [b,257,1],[b,37,1] -> [b,257,1]

            recons_loss, KL_loss = loss_function(recon_batch, gt, mu, logvar)
            loss = recons_loss + args.beta* KL_loss
            total_recons_loss += recons_loss.item()
            total_kl_loss += KL_loss.item()
            total_pred += keypoint.shape[0]
            # 反向传播
            loss.backward()
            # 参数更新
            optimizer.step()    
        total_recons_loss = total_recons_loss / total_pred    
        total_kl_loss = total_kl_loss / total_pred    
        # 打印loss
        print(f'====> Epoch: {epoch} recons loss: {total_recons_loss} kl_loss: {total_kl_loss}')

    @torch.no_grad()
    def infer(self,args, model, dataloader,device, epoch):
        """测试模型的metrics: label error, smoothness"""
        model.eval()

        total_parsec_loss = [[0]*3 for _ in range(11)]
        total_smooth = []
        total_pred = 0  # 总共的样本数量
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
            recon_batch = model.sample(condition) # [b,20],[b,37,1] -> [b,257,1]
            total_pred += recon_batch.shape[0]

            # 统计一下物理量之间的误差
            for idx in range(recon_batch.shape[0]): # 测第一个即可
                # 给他们拼接同一个x坐标
                source = recon_batch[idx][:,0].detach().cpu().numpy() # [257]
                target = gt[idx][:,0].detach().cpu().numpy() # [257]

                # 需要check 一下为啥x直接就是numpy格式
                source = np.stack([x,source],axis=1)
                total_smooth.append(calculate_smoothness(source))

                target = np.stack([x,target],axis=1)
                source_parsec = Fit_airfoil(source).parsec_features
                target_parsec = Fit_airfoil(target).parsec_features
                for i in range(11):
                    total_parsec_loss[i][0] += abs(source_parsec[i]-target_parsec[i]) # 绝对误差
                    total_parsec_loss[i][1] += abs(source_parsec[i]-target_parsec[i])/(abs(target_parsec[i])+1e-9) # 相对误差
                    total_parsec_loss[i][2] += abs(target_parsec[i]) # 真实值的绝对值
                # if idx % 100 == 0:
                #   vis_airfoil2(source,target,epoch+idx,dir_name=args.log_dir,sample_type='cvae')
            
        avg_parsec_loss = [(x/total_pred,y/total_pred,z/total_pred) for x,y,z in total_parsec_loss]
        avg_parsec_loss_sci = [f"{x:.2e}" for x, y, z in avg_parsec_loss]

        smoothness = np.nanmean(total_smooth,0)

        
        print(f"infer——epoch: {epoch}, avg_parsec_loss: {' & '.join(avg_parsec_loss_sci)}")
        print(f"infer——epoch: {epoch}, smoothness: {smoothness}")
        # 保存评测结果
        with open(f'{args.log_dir}/infer_result.txt','a') as f:
            f.write(f"infer——epoch: {epoch}, avg_parsec_loss: {' & '.join(avg_parsec_loss_sci)}\n")
            f.write(f"infer——epoch: {epoch}, smoothness: {smoothness}")
    
    @torch.no_grad()
    def infer_diversity(self,args, model, dataloader,device, epoch):
        """测试模型的diversity score"""
        model.eval()

        total_div_score = []

        test_loader = tqdm(dataloader)
        for _,data in enumerate(test_loader):
            for i in range(data['keypoint'].shape[0]):
              keypoint = data['keypoint'][i,:,1:2] # [26,1]
              physics = data['params'][i] # [11]
              physics = physics.unsqueeze(-1) # [11,1]
              condition = torch.cat([physics,keypoint],dim=0)
              condition = condition.repeat(10,1,1)
              condition = condition.to(device)
              recon_batch = model.sample(condition) #  [10,37,1] -> [1,257,1]
              recon_batch = recon_batch.cpu().numpy()
              total_div_score.append(cal_diversity_score(recon_batch))
            break # 只跑一个batch_size
        pkvae_diver = np.nanmean(total_div_score,0)
        print(f"infer——epoch: {epoch}, pkvae_diver: {pkvae_diver}")
        with open(f'{args.log_dir}/eval_result.txt','a') as f:
            f.write(f"infer——epoch: {epoch}, pkvae_diver: {pkvae_diver}\n")

    def main(self,args):
        """Run main training/evaluation pipeline."""

        # 单卡训练
        model = self.get_model(args)
        device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        model.to(device)

        train_loader, test_loader = self.get_loaders(args) 
        optimizer = self.get_optimizer(args,model)
        # Scheduler https://arxiv.org/pdf/1812.01187.pdf
        lf = lambda x: ((1 + math.cos(x * math.pi / args.max_epoch)) / 2) * (1 - args.lrf) + args.lrf  # cosine
        scheduler = lr_scheduler.LambdaLR(optimizer, lr_lambda=lf)
      
 
        # Check for a checkpoint
        if len(args.checkpoint_path)>0:
            assert os.path.isfile(args.checkpoint_path)
            load_checkpoint(args, model, optimizer, scheduler)
        
        # set log dir
        os.makedirs(args.log_dir, exist_ok=True)
        for epoch in range(args.start_epoch,args.max_epoch+1):
            # # train
            # self.train_one_epoch(args=args,
            #                      model=model,
            #                      optimizer=optimizer,
            #                      dataloader=train_loader,
            #                      device=device,
            #                      epoch=epoch
            #                      )
            # scheduler.step()

            # save model and validate
            if epoch % args.val_freq == 0:
                # save_checkpoint(args, epoch, model, optimizer, scheduler)
                # # print("test begin.......")
                # self.infer(
                #     args=args,
                #     model=model,
                #     dataloader=test_loader,
                #     device=device, 
                #     epoch=epoch, 
                #     )
                self.infer_diversity(
                    args=args,
                    model=model,
                    dataloader=test_loader,
                    device=device, 
                    epoch=epoch, 
                    )
                    

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




'''
python train_softvae.py --log_dir logs/soft_vae_afbench  --max_epoch 201 --val_freq 100 --save_freq 100 --checkpoint_path logs/soft_vae_afbench/ckpt_epoch_200.pth --batch_size 100
'''
       