import os
import torch
import torch.nn as nn
from tqdm import tqdm 
from torch.utils.data import DataLoader
import torch.optim as optim 
import torch.nn.functional as F
import torch.optim.lr_scheduler as lr_scheduler
from models.bezier_gan import Generator,Discriminator
from dataload import AirFoilMixParsec, Fit_airfoil
import math 
from utils import vis_airfoil2
import random
import argparse
import numpy as np
from matplotlib import pyplot as plt


os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
EPSILON = 1e-7

def write(file_path, data):
    with open(file_path,'w') as f:
        for x,y in data:
            f.write(f'{x} {y}\n')

def parse_option():
    """Parse cmd arguments."""
    parser = argparse.ArgumentParser()
    # Data
    parser.add_argument('--batch_size', type=int, default=64,
                        help='Batch Size during training')
    parser.add_argument('--latent_size', type=int, default=10,
                        help='Batch Size during training')
    parser.add_argument('--num_workers',type=int,default=4)
    parser.add_argument('--device', type=str, default='cuda:0')

    # Training
    parser.add_argument('--beta', default=1, type=float, help='beta parameter for KL-term in original beta-VAE')
    parser.add_argument('--start_epoch', type=int, default=1)
    parser.add_argument('--max_epoch', type=int, default=501)
    parser.add_argument('--weight_decay', type=float, default=0.0005)
    parser.add_argument("--lr", default=3e-4, type=float)
    parser.add_argument('--lrf', type=float, default=0.01)
    parser.add_argument('--lr-scheduler', type=str, default='step',
                          choices=["step", "cosine"])
    parser.add_argument('--lr_decay_epochs', type=int, default=[50, 75],
                        nargs='+', help='when to decay lr, can be a list')
    parser.add_argument('--lr_decay_rate', type=float, default=0.1,
                        help='for step scheduler. decay rate for lr')
    parser.add_argument('--warmup-epoch', type=int, default=-1)
    parser.add_argument('--warmup-multiplier', type=int, default=100)
    parser.add_argument('--latent_dim', type=int, default=3)
    parser.add_argument('--noise_dim', type=int, default=10)
    parser.add_argument('--n_points', type=int, default=257)

    # io
    parser.add_argument('--checkpoint_path_d', default='',help='Model checkpoint path') # logs/cvae_gan/d_ckpt_epoch_500.pth
    parser.add_argument('--checkpoint_path_g', default='',help='Model checkpoint path') # logs/cvae_gan/g_ckpt_epoch_500.pth
    parser.add_argument('--log_dir', default=f'logs/cgan',
                        help='Dump dir to save model checkpoint & experiment log')
    parser.add_argument('--val_freq', type=int, default=100)  # epoch-wise
    parser.add_argument('--save_freq', type=int, default=500)  # epoch-wise
    

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
        train_dataset = AirFoilMixParsec(split='train',dataset_names=['interpolated_uiuc'])  
        val_dataset = AirFoilMixParsec(split='val',dataset_names=['interpolated_uiuc']) 
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
        # 创建对象
        D = Discriminator(latent_dim=args.latent_dim, n_points=args.n_points)
        G = Generator(latent_dim=args.latent_dim, noise_dim=args.noise_dim,n_points=args.n_points)
        return D,G

    @staticmethod
    def get_optimizer(args,D,G):
        params = [p for p in D.parameters() if p.requires_grad]
        d_optimizer = optim.AdamW(params,
                              lr = args.lr,
                              weight_decay=args.weight_decay)
        
        params = [p for p in G.parameters() if p.requires_grad]
        g_optimizer = optim.AdamW(params,
                              lr = args.lr,
                              weight_decay=args.weight_decay)
        return d_optimizer, g_optimizer
    


    def train_one_epoch(self,args, D,G,d_optimizer,g_optimizer,dataloader,device,epoch):
        """训练一个epoch"""
        D.train()  # set model to training mode
        G.train()
        bounds = (0.0, 1.0)
        for _,data in enumerate(tqdm(dataloader)):
            X_real = data['gt'].unsqueeze(dim=-1) # [b,257,2,1]
            batch_size = X_real.shape[0]
            X_real = X_real.to(device)
            # train discriminator:
            # train d_real
            y_latent = np.random.uniform(low=bounds[0], high=bounds[1], size=(batch_size, args.latent_dim))
            noise = np.random.normal(scale=0.5, size=(batch_size, args.noise_dim))
            y_latent = torch.from_numpy(y_latent).to(device)
            y_latent = y_latent.float()
            noise = torch.from_numpy(noise).to(device)
            noise = noise.float()
            d_real, _ = D(X_real)
            BCEWithLogitsLoss = nn.BCEWithLogitsLoss()
            d_loss_real = torch.mean(BCEWithLogitsLoss(d_real, torch.ones_like(d_real)))

            # train d_fake
            x_fake_train, cp_train, w_train, ub_train, db_train = G(y_latent, noise)
            d_fake, q_fake_train = D(x_fake_train.detach())
            d_loss_fake = torch.mean(BCEWithLogitsLoss(d_fake, torch.zeros_like(d_fake)))
            q_mean = q_fake_train[:, 0, :]
            q_logstd = q_fake_train[:, 1, :]
            q_target = y_latent
            epsilon = (q_target - q_mean) / (torch.exp(q_logstd) + EPSILON)
            q_loss = q_logstd + 0.5 * torch.square(epsilon)
            q_loss = torch.mean(q_loss)
            d_train_real_loss = d_loss_real
            d_train_fake_loss = d_loss_fake + q_loss

            D_loss = d_train_real_loss.item() + d_train_fake_loss.item()

            d_optimizer.zero_grad()
            d_train_real_loss.backward()
            d_train_fake_loss.backward()
            
            d_optimizer.step()
            # print("training Discriminator. D real loss:", d_train_real_loss.item(), "D fake loss:", d_train_fake_loss.item())

            # train g_loss
            y_latent = np.random.uniform(low=bounds[0], high=bounds[1], size=(batch_size, args.latent_dim))
            noise = np.random.normal(scale=0.5, size=(batch_size, args.noise_dim))
            y_latent = torch.from_numpy(y_latent).to(device)
            y_latent = y_latent.float()
            noise = torch.from_numpy(noise).to(device)
            noise = noise.float()
            x_fake_train, cp_train, w_train, ub_train, db_train = G(y_latent, noise)
            d_fake, q_fake_train = D(x_fake_train)
            g_loss = torch.mean(BCEWithLogitsLoss(d_fake, torch.ones_like(d_fake)))

            # Regularization for w, cp, a, and b
            r_w_loss = torch.mean(w_train[:,1:-1], dim=(1,2))
            cp_dist = torch.norm(cp_train[:, 1:] - cp_train[:, :-1], dim=-1)
            r_cp_loss = torch.mean(cp_dist, dim=-1)
            r_cp_loss1 = torch.max(cp_dist, dim=-1)[0]
            ends = cp_train[:, 0] - cp_train[:, -1]
            r_ends_loss = torch.norm(ends, dim=-1) + torch.maximum(torch.tensor(0.0).to(device), -10 * ends[:, 1])
            r_db_loss = torch.mean(db_train * torch.log(db_train), dim=-1)
            r_loss = r_w_loss + r_cp_loss + 0 * r_cp_loss1 + r_ends_loss + 0 * r_db_loss
            r_loss = torch.mean(r_loss)
            q_mean = q_fake_train[:, 0, :]
            q_logstd = q_fake_train[:, 1, :]
            q_target = y_latent
            epsilon = (q_target - q_mean) / (torch.exp(q_logstd) + EPSILON)
            q_loss = q_logstd + 0.5 * torch.square(epsilon)
            q_loss = torch.mean(q_loss)
            # Gaussian loss for Q
            G_loss = g_loss + 10*r_loss + q_loss
            g_optimizer.zero_grad()
            G_loss.backward()
            g_optimizer.step()

        # 打印loss
        print(f'====> Epoch: {epoch} generator loss: {G_loss} discriminator: {D_loss}')
        with open(f'{args.log_dir}/train_result.txt','a') as f:
            f.write(f"train——epoch: {epoch}, generator loss: {G_loss}, discriminator: {D_loss}\n")

    @torch.no_grad()
    def data_gen(self, args, generator, batch_size):
        curIdx = 0
        generator.eval()
        os.makedirs(f'data/airfoil/bezier_gen',exist_ok=True)
        for _ in tqdm(range(10000//64)):
          bounds = (0.0, 1.0)
          y_latent = np.random.uniform(low=bounds[0], high=bounds[1], size=(batch_size, args.latent_dim))
          noise = np.random.normal(scale=0.5, size=(batch_size, args.noise_dim))
          y_latent = torch.from_numpy(y_latent).to(args.device)
          y_latent = y_latent.float()
          noise = torch.from_numpy(noise).to(args.device)
          noise = noise.float()
          x_fake_train, cp_train, w_train, ub_train, db_train = generator(y_latent, noise)
          x_fake_train = x_fake_train.squeeze(dim=-1)
          airfoil = x_fake_train.detach().cpu().numpy() # [64,257,2]
          # 将airfoil写入到文件
          for idx in range(airfoil.shape[0]):
              data = airfoil[idx]
              file_path = f'data/airfoil/bezier_gen/{curIdx:05d}.dat'
              write(file_path,data)
              curIdx += 1
        # # Reshape the array to have dimensions [8, 8, 257, 2]
        # airfoils = airfoil.reshape(8, 8, 257, 2)

        # # Create a figure with subplots
        # fig, axs = plt.subplots(8, 8, figsize=(12, 12))

        # # Iterate over each row and column of the grid
        # for i in range(8):
        #     for j in range(8):
        #         # Get the airfoil coordinates for the current grid position
        #         airfoil = airfoils[i, j]

        #         # Plot the airfoil on the corresponding subplot
        #         axs[i, j].plot(airfoil[:, 0], airfoil[:, 1])
        #         axs[i, j].set_aspect('equal', 'box')

        #         # Remove ticks and labels for a cleaner plot
        #         axs[i, j].axis('off')

        # # Adjust the spacing between subplots
        # fig.tight_layout()

        # # Save the figure
        # plt.savefig(f'{args.log_dir}/airfoil_grid_{epoch}.png')
        # plt.close()
    
    @torch.no_grad()
    def infer_old(self,args, model, dataloader,device, epoch):
        """验证一个epoch"""
        model.eval()

        total_parsec_loss = [[0]*3 for _ in range(11)]

        
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
                target = np.stack([x,target],axis=1)
                source_parsec = Fit_airfoil(source).parsec_features
                target_parsec = Fit_airfoil(target).parsec_features
                for i in range(11):
                    total_parsec_loss[i][0] += abs(source_parsec[i]-target_parsec[i]) # 绝对误差
                    total_parsec_loss[i][1] += abs(source_parsec[i]-target_parsec[i])/(abs(target_parsec[i])+1e-9) # 相对误差
                    total_parsec_loss[i][2] += abs(target_parsec[i]) # 真实值的绝对值
                if idx < 5:
                  vis_airfoil2(source,target,epoch+idx,dir_name=args.log_dir,sample_type='cvae')
        accuracy = correct_pred / total_pred
        avg_loss = total_loss / total_pred
        avg_parsec_loss = [(x/total_pred,y/total_pred,z/total_pred) for x,y,z in total_parsec_loss]
        # 将avg_parsec_loss中的每个元素转换为科学计数法，保留两位有效数字
        avg_parsec_loss_sci = [f"{x:.2e}" for x, y, z in avg_parsec_loss]

        print(f"infer——epoch: {epoch}, accuracy: {accuracy}, avg_loss: {avg_loss}")
        print(f"infer——epoch: {epoch}, avg_parsec_loss: {' & '.join(avg_parsec_loss_sci)}")
        # 保存评测结果
        with open(f'{args.log_dir}/infer_result.txt','a') as f:
            f.write(f"infer——epoch: {epoch}, accuracy: {accuracy}, keypoint_loss: {avg_loss:.2e}\n")
            f.write(f"infer——epoch: {epoch}, avg_parsec_loss: {' & '.join(avg_parsec_loss_sci)}\n")


    def main(self,args):
        """Run main training/evaluation pipeline."""
        
        # 单卡训练
        D,G = self.get_model(args)
        # device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        device = args.device
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

        # self.data_gen(args=args, generator=G, batch_size=args.batch_size)
        # # set log dir
        os.makedirs(args.log_dir, exist_ok=True)
        for epoch in range(args.start_epoch,args.max_epoch+1):
            # # train
            # self.train_one_epoch(args=args,
            #                      D=D,
            #                      G=G,
            #                      d_optimizer=d_optimizer,
            #                      g_optimizer=g_optimizer,
            #                      dataloader=train_loader,
            #                      device=device,
            #                      epoch=epoch
            #                      )
            # d_scheduler.step()
            # g_scheduler.step()
            # save model and validate
            # args.val_freq = 1
            if epoch % args.val_freq == 0:
                save_checkpoint(args, epoch, D,G,d_optimizer,g_optimizer)
                print("Validation begin.......")
                self.infer(args=args, generator=G, batch_size=args.batch_size,epoch=epoch)
                # self.infer(
                #     args=args,
                #     model=G,
                #     dataloader=val_loader,
                #     device=device, 
                #     epoch=epoch, 
                #     )
                 
           
          
                

if __name__ == '__main__':
    opt = parse_option()
    # Fix random seed for reproducibility
    seed = 42
    torch.manual_seed(seed)
    random.seed(seed)


    # torch.backends.cudnn.enabled = True
    # # 启用cudnn的自动优化模式
    # torch.backends.cudnn.benchmark = True
    # # 设置cudnn的确定性模式，pytorch确保每次运行结果的输出都保持一致
    # torch.backends.cudnn.deterministic = True

    trainer = Trainer()
    trainer.main(opt)


''''
python train_bezier_gan.py --log_dir logs/bezier_gan --val_freq 100 --save_freq 500 --device cuda:0 --checkpoint_path_d logs/bezier_gan/d_ckpt_epoch_500.pth --checkpoint_path_g logs/bezier_gan/g_ckpt_epoch_500.pth
'''