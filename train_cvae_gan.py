import os
import torch
import torch.nn as nn
from tqdm import tqdm 
from torch.utils.data import DataLoader
import torch.optim as optim 
import torch.nn.functional as F
import torch.optim.lr_scheduler as lr_scheduler
from models import CVAE_GAN, Discriminator
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
    parser.add_argument('--checkpoint_path_d', default='',help='Model checkpoint path') # logs/cvae_gan/d_ckpt_epoch_500.pth
    parser.add_argument('--checkpoint_path_g', default='',help='Model checkpoint path') # logs/cvae_gan/g_ckpt_epoch_500.pth
    parser.add_argument('--log_dir', default=f'logs/cvae_gan',
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
        D = Discriminator()
        G = CVAE_GAN()
        return D,G

    @staticmethod
    def get_optimizer(args,D,G):
        params = [p for p in D.parameters() if p.requires_grad]
        d_optimizer = optim.Adam(params,
                              lr = args.lr,
                              weight_decay=args.weight_decay)
        
        params = [p for p in G.parameters() if p.requires_grad]
        g_optimizer = optim.Adam(params,
                              lr = args.lr,
                              weight_decay=args.weight_decay)
        return d_optimizer, g_optimizer
    


    def train_one_epoch(self,args, D,G,d_optimizer,g_optimizer,dataloader,device,epoch):
        """训练一个epoch"""
        D.train()  # set model to training mode
        G.train()
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

            ## (1) Update D network: maximize log(D(x)) + log(1 - D(G(z)))
            for _ in range(5):
              num = gt.shape[0]
              d_gt = gt.reshape(num,  -1)   
              real_airfoil = d_gt.to(device) # 将tensor变成Variable放入计算图中
              real_label = torch.ones(num).to(device)  # 定义真实的airfoil label为1
              fake_label = torch.zeros(num).to(device) # 定义假的airfoil 的label为0

              # 计算真实airfoil的损失
              real_out = D(real_airfoil)  # 将真实airfoil放入判别器中
              real_scores = real_out  # 得到真实图片的判别值，输出的值越接近1越好

              # 计算假的airfoil的损失
              recon_batch = G.sample(condition) # [b,257],[b,37] -> [b,257]
              fake_airfoil = recon_batch.reshape(num,-1) 
              fake_out = D(fake_airfoil.detach())  # 判别器判断假的airfoil
              d_loss = torch.mean(fake_out)-torch.mean(real_out)  # 得到airfoil的loss
              fake_scores = fake_out  # 得到假airfoil的判别值，对于判别器来说，假airfoil的损失越接近0越好

              # 损失函数和优化
              # d_loss = d_loss_real + d_loss_fake  # 损失包括判真损失和判假损失
              d_optimizer.zero_grad()  # 在反向传播之前，先将梯度归0
              d_loss.backward()  # 将误差反向传播
              d_optimizer.step()  # 更新参数
              # weight Clipping WGAN
              for layer in D.dis:
                  if (layer.__class__.__name__ == 'Linear'):
                      layer.weight.requires_grad = False
                      layer.weight.clamp_(-0.005, 0.005)
                      layer.weight.requires_grad = True


            # (2) Update G network which is the decoder of VAE
            recon_batch, mu, logvar = G(gt,condition) # [b,257],[b,37] -> [b,257]
            G.zero_grad()
            recons_loss, KL_loss = loss_function(recon_batch, gt, mu, logvar)
            loss = recons_loss + args.beta* KL_loss
            total_recons_loss += recons_loss.item()
            total_kl_loss += KL_loss.item()
            total_pred += keypoint.shape[0]
            loss.backward(retain_graph=True)
            g_optimizer.step()

            # (3) Update G network: maximize log(D(G(z)))
            G.zero_grad()
            fake_airfoil = recon_batch.reshape(num,-1) 
            output = D(fake_airfoil.detach()).squeeze(1)  
            g_loss = torch.mean(-output)  
            g_loss.backward()  # 进行反向传播
            g_optimizer.step()  # .step()一般用在反向传播后面,用于更新生成网络的参数  
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
                if idx % 100 == 0:
                  vis_airfoil2(source,target,epoch+idx,dir_name=args.log_dir,sample_type='cvae')
            
        avg_parsec_loss = [(x/total_pred,y/total_pred,z/total_pred) for x,y,z in total_parsec_loss]
        avg_parsec_loss_sci = [f"{x:.2e}" for x, y, z in avg_parsec_loss]

        smoothness = np.nanmean(total_smooth,0)

        
        print(f"infer——epoch: {epoch}, avg_parsec_loss: {' & '.join(avg_parsec_loss_sci)}")
        print(f"infer——epoch: {epoch}, smoothness: {smoothness}")
        # 保存评测结果
        with open(f'{args.log_dir}/infer_result.txt','a') as f:
            f.write(f"infer——epoch: {epoch}, avg_parsec_loss: {' & '.join(avg_parsec_loss_sci)}\n")
            f.write(f"infer——epoch: {epoch}, smoothness: {smoothness}\n")

    def main(self,args):
        """Run main training/evaluation pipeline."""

        # 单卡训练
        D,G = self.get_model(args)
        device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        D.to(device)
        G.to(device)

        train_loader, val_loader = self.get_loaders(args) 
        d_optimizer, g_optimizer = self.get_optimizer(args,D,G)
        lf = lambda x: ((1 + math.cos(x * math.pi / args.max_epoch)) / 2) * (1 - args.lrf) + args.lrf  # cosine
        d_scheduler = lr_scheduler.LambdaLR(d_optimizer, lr_lambda=lf)
        g_scheduler = lr_scheduler.LambdaLR(g_optimizer, lr_lambda=lf)

   
        # Check for a checkpoint
        if len(args.checkpoint_path_g)>0 and len(args.checkpoint_path_d)>0:
            assert os.path.isfile(args.checkpoint_path_g)
            load_checkpoint(args, D, G, d_optimizer,g_optimizer)
        
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
                                 epoch=epoch
                                 )
            d_scheduler.step()
            g_scheduler.step()
            # save model and validate
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
python train_cvae_gan.py --log_dir logs/cvae_gan_super
python train_cvae_gan.py --log_dir logs/cvae_gan_afbench --max_epoch 201 --val_freq 100 --save_freq 100
python train_cvae_gan.py --checkpoint_path_d logs/cvae_gan/d_ckpt_epoch_1000.pth --checkpoint_path_g logs/cvae_gan/g_ckpt_epoch_1000.pth  
'''