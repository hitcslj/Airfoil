import os
import torch
import torch.nn as nn
from tqdm import tqdm 
from torch.utils.data import DataLoader
import torch.optim as optim 
import torch.nn.functional as F
import torch.optim.lr_scheduler as lr_scheduler
from models import Generator,CDiscriminator
from dataload import AirFoilMixParsec, Fit_airfoil
import math 
from utils import vis_airfoil2
import random
import argparse
import numpy as np
from utils import calculate_smoothness


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
    parser.add_argument("--lr", default=1e-4, type=float)
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
    parser.add_argument('--checkpoint_path_d', default='',help='Model checkpoint path') # logs/cvae_gan/d_ckpt_epoch_1000.pth
    parser.add_argument('--checkpoint_path_g', default='',help='Model checkpoint path') # logs/cvae_gan/g_ckpt_epoch_500.pth
    parser.add_argument('--log_dir', default=f'logs/cgan',
                        help='Dump dir to save model checkpoint & experiment log')
    parser.add_argument('--val_freq', type=int, default=1000)  # epoch-wise
    parser.add_argument('--save_freq', type=int, default=1000)  # epoch-wise
    

    # 评测指标相关
    parser.add_argument('--distance_threshold', type=float, default=0.01) # xy点的距离小于该值，被认为是预测正确的点
    parser.add_argument('--threshold_ratio', type=float, default=0.75) # 200个点中，预测正确的点超过一定的比例，被认为是预测正确的样本
   
    args, _ = parser.parse_known_args()


    return args

# BRIEF load checkpoint.
def load_checkpoint(args, D, G, d_optimizer,g_optimizer):
    """Load from checkpoint."""
    print("=> loading checkpoint '{}'".format(args.checkpoint_path_d))

    checkpoint = torch.load(args.checkpoint_path_d, map_location='cpu')
    try:
        args.start_epoch = int(checkpoint['epoch'])
    except Exception:
        args.start_epoch = 1
    D.load_state_dict(checkpoint['model'], strict=True)
    d_optimizer.load_state_dict(checkpoint['optimizer'])

    print("=> loading checkpoint '{}'".format(args.checkpoint_path_g))
    checkpoint = torch.load(args.checkpoint_path_g, map_location='cpu')
    try:
        args.start_epoch = int(checkpoint['epoch'])
    except Exception:
        args.start_epoch = 1
    G.load_state_dict(checkpoint['model'], strict=True)
    g_optimizer.load_state_dict(checkpoint['optimizer'])

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
def save_checkpoint(args, epoch, D, G, d_optimizer, g_optimizer):
    """Save checkpoint if requested."""
    if epoch % args.save_freq == 0:
        state = {
            'model': D.state_dict(),
            'optimizer': d_optimizer.state_dict(),
            'epoch': epoch
        }
        os.makedirs(args.log_dir, exist_ok=True)
        spath = os.path.join(args.log_dir, f'd_ckpt_epoch_{epoch}.pth')
        state['save_path'] = spath
        torch.save(state, spath)
        print("Saved in {}".format(spath))

        state = {
            'model': G.state_dict(),
            'optimizer': g_optimizer.state_dict(),
            'epoch': epoch
        }
        os.makedirs(args.log_dir, exist_ok=True)
        spath = os.path.join(args.log_dir, f'g_ckpt_epoch_{epoch}.pth')
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
        # 创建对象
        D = CDiscriminator()
        G = Generator()
        return D,G

    @staticmethod
    def get_optimizer(args,D,G):
        d_params = [p for p in D.parameters()]
        d_optimizer = optim.Adam(d_params,
                              lr = 4*args.lr,
                              weight_decay=args.weight_decay)
        
        g_params = [p for p in G.parameters()]
        g_optimizer = optim.Adam(g_params,
                              lr = args.lr,
                              weight_decay=args.weight_decay)
        return d_optimizer, g_optimizer
    


    def train_one_epoch(self,args, D,G,d_optimizer,g_optimizer,dataloader,device,epoch,adversarial_criterion):
        """训练一个epoch"""
        D.train()  # set model to training mode
        G.train()
        total_g_loss = total_d_loss = 0
        total_pred = 0
        for _,data in enumerate(tqdm(dataloader)):
            keypoint = data['keypoint'][:,:,1:2] # [b,26,1]
            gt = data['gt'][:,:,1:2] # [b,257,1]
            physics = data['params'] # [b,11]
            physics = physics.unsqueeze(-1) # [b,11,1]
            condition = torch.cat([physics,keypoint],dim=1)
            condition = condition.to(device)
            gt = gt.to(device)
            batch_size = gt.shape[0]

            real_label = torch.full((batch_size, 1), 1, dtype=gt.dtype).to(device)
            fake_label = torch.full((batch_size, 1), 0, dtype=gt.dtype).to(device)

            total_pred += 1
            ## (1) Update D network: maximize log(D(x)) + log(1 - D(G(z)))
            d_optimizer.zero_grad()

            real_airfoil = gt.reshape(batch_size,-1).to(device)  
            real_out = D(real_airfoil, condition)  
            d_loss_real = adversarial_criterion(real_out, real_label)
            d_loss_real.backward()

            noise = torch.randn(batch_size, args.latent_size).to(device)  # 生成随机噪声
            fake_airfoil = G(noise,condition).reshape(batch_size,-1) 
            fake_out = D(fake_airfoil.detach(),condition)  
            d_loss_fake = adversarial_criterion(fake_out, fake_label)
            d_loss_fake.backward()

            d_loss = d_loss_real + d_loss_fake
            total_d_loss += d_loss.item()
            d_optimizer.step()  # 更新参数

            # (2) Update G network: maximize log(D(G(z)))
            g_optimizer.zero_grad()
            fake_out = D(fake_airfoil,condition)  
            g_loss = adversarial_criterion(fake_out, real_label)
            total_g_loss += g_loss.item()
            g_loss.backward()
            g_optimizer.step()  


        total_g_loss = total_g_loss / total_pred    
        total_d_loss = total_d_loss / total_pred    
        # 打印loss
        print(f'====> Epoch: {epoch} generator loss: {total_g_loss} discriminator: {total_d_loss}')
  

    @torch.no_grad()
    def infer(self,args, model, dataloader,device, epoch):
        """验证一个epoch"""
        model.eval()

        total_parsec_loss = [[0]*3 for _ in range(11)]

        
        correct_pred = 0  # 预测正确的样本数量
        total_pred = 0  # 总共的样本数量
        total_loss = 0.0
        total_smoothness = []
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
            num = gt.shape[0]
            noise = torch.randn(num, args.latent_size).to(device)  # 生成随机噪声
            recon_batch = model(noise,condition)  

            total_pred += keypoint.shape[0]

            loss = nn.MSELoss()(recon_batch[:,::10], gt[:,::10])
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
                total_smoothness.append(calculate_smoothness(source))

                target = np.stack([x,target],axis=1)
                source_parsec = Fit_airfoil(source).parsec_features
                target_parsec = Fit_airfoil(target).parsec_features
                for i in range(11):
                    total_parsec_loss[i][0] += abs(source_parsec[i]-target_parsec[i]) # 绝对误差
                    total_parsec_loss[i][1] += abs(source_parsec[i]-target_parsec[i])/(abs(target_parsec[i])+1e-9) # 相对误差
                    total_parsec_loss[i][2] += abs(target_parsec[i]) # 真实值的绝对值
                if idx < 10:
                  vis_airfoil2(source,target,epoch+idx,dir_name=args.log_dir,sample_type='cgan')
        accuracy = correct_pred / total_pred
        avg_loss = total_loss / total_pred
        avg_parsec_loss = [(x/total_pred,y/total_pred,z/total_pred) for x,y,z in total_parsec_loss]
        # 将avg_parsec_loss中的每个元素转换为科学计数法，保留两位有效数字
        avg_parsec_loss_sci = [f"{x:.2e}" for x, y, z in avg_parsec_loss]
        smoothness = np.nanmean(total_smoothness,0)

        print(f"infer——epoch: {epoch}, accuracy: {accuracy}, avg_loss: {avg_loss}")
        print(f"infer——epoch: {epoch}, avg_parsec_loss: {' & '.join(avg_parsec_loss_sci)}")
        print(f"infer——epoch: {epoch}, smoothness: {smoothness}")
        # 保存评测结果
        with open(f'{args.log_dir}/infer_result.txt','a') as f:
            f.write(f"infer——epoch: {epoch}, accuracy: {accuracy}, keypoint_loss: {avg_loss:.2e}\n")
            f.write(f"infer——epoch: {epoch}, avg_parsec_loss: {' & '.join(avg_parsec_loss_sci)}\n")
            f.write(f"infer——epoch: {epoch}, smoothness: {smoothness}")


    def main(self,args):
        """Run main training/evaluation pipeline."""

        # 单卡训练
        D,G = self.get_model(args)
        device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        D.to(device)
        G.to(device)

        train_loader, val_loader = self.get_loaders(args) 
        d_optimizer, g_optimizer = self.get_optimizer(args,D,G)
        adversarial_criterion = nn.MSELoss().to(device)
   
        # Check for a checkpoint
        if len(args.checkpoint_path_g)>0 and len(args.checkpoint_path_d)>0:
            assert os.path.isfile(args.checkpoint_path_g)
            load_checkpoint(args, D, G, d_optimizer,g_optimizer)

        lf = lambda x: ((1 + math.cos(x * math.pi / args.max_epoch)) / 2) * (1 - args.lrf) + args.lrf  # cosine
        d_scheduler = lr_scheduler.LambdaLR(d_optimizer, lr_lambda=lf)
        g_scheduler = lr_scheduler.LambdaLR(g_optimizer, lr_lambda=lf)
        # set log dir
        os.makedirs(args.log_dir, exist_ok=True)
        for epoch in range(args.start_epoch,args.max_epoch+1):
            # # train
            self.train_one_epoch(args=args,
                                 D=D,
                                 G=G,
                                 d_optimizer=d_optimizer,
                                 g_optimizer=g_optimizer,
                                 dataloader=train_loader,
                                 device=device,
                                 epoch=epoch,
                                 adversarial_criterion=adversarial_criterion,
                                 )
            d_scheduler.step()
            g_scheduler.step()
            # save model and validate
            # args.val_freq = 1
            if epoch % args.val_freq == 0:
                save_checkpoint(args, epoch, D,G,d_optimizer,g_optimizer)
                print("Validation begin.......")
                self.infer(
                    args=args,
                    model=G,
                    dataloader=val_loader,
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
python train_cgan.py --log_dir logs/cgan_super
python train_cgan.py --log_dir logs/cgan_afbench --max_epoch 201 --val_freq 100 --save_freq 100
'''
