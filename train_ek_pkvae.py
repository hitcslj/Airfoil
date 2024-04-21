import argparse
import os
import torch
import torch.nn as nn
from tqdm import tqdm 
from torch.utils.data import DataLoader
import torch.optim as optim 
 
import torch.optim.lr_scheduler as lr_scheduler
from models import EK_VAE,PK_VAE,EK_PKVAE
from dataload import EditingMixDataset
import math 
from utils import vis_airfoil2
from dataload.parsec_direct import Fit_airfoil
import numpy as np
    
def parse_option():
    """Parse cmd arguments."""
    parser = argparse.ArgumentParser()
    # Data
    parser.add_argument('--batch_size', type=int, default=512,
                        help='Batch Size during training')
    parser.add_argument('--num_workers',type=int,default=4)
    # Training
    parser.add_argument('--start_epoch', type=int, default=1)
    parser.add_argument('--max_epoch', type=int, default=1000)
    parser.add_argument('--weight_decay', type=float, default=0.0005)
    parser.add_argument("--lr", default=1e-3, type=float)
    parser.add_argument('--lrf', type=float, default=0.01)
    parser.add_argument('--lr-scheduler', type=str, default='step',
                          choices=["step", "cosine"])
    parser.add_argument('--lr_decay_epochs', type=int, default=[20, 50],
                        nargs='+', help='when to decay lr, can be a list')
    parser.add_argument('--lr_decay_rate', type=float, default=0.1,
                        help='for step scheduler. decay rate for lr')
    parser.add_argument('--warmup-epoch', type=int, default=-1)
    parser.add_argument('--warmup-multiplier', type=int, default=100)

    # io
    parser.add_argument('--checkpoint_path_A', default='logs/ek_vae/ckpt_epoch_1000.pth',help='Model checkpoint path')

    parser.add_argument('--checkpoint_path_B', default='logs/pk_vae/ckpt_epoch_2000.pth',help='Model checkpoint path')

    parser.add_argument('--log_dir', default='logs/ek_pkvae',
                        help='Dump dir to save model checkpoint')
    parser.add_argument('--val_freq', type=int, default=100)  # epoch-wise
    parser.add_argument('--save_freq', type=int, default=1000)  # epoch-wise
    

    # 评测指标相关
    parser.add_argument('--distance_threshold', type=float, default=0.01) # xy点的距离小于该值，被认为是预测正确的点
    parser.add_argument('--threshold_ratio', type=float, default=0.75) # 200个点中，预测正确的点超过一定的比例，被认为是预测正确的样本
   
    args, _ = parser.parse_known_args()


    return args

# BRIEF load checkpoint.
def load_checkpoint(args,path, model):
    """Load from checkpoint."""
    print("=> loading checkpoint {}".format(path))

    checkpoint = torch.load(path, map_location='cpu')
    model.load_state_dict(checkpoint['model'], strict=True)
    # optimizer.load_state_dict(checkpoint['optimizer'])
    # scheduler.load_state_dict(checkpoint['scheduler'])

    print("=> loaded successfully  (epoch {})".format(
        checkpoint['epoch']
    ))

    del checkpoint
    torch.cuda.empty_cache()

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
        train_dataset = EditingMixDataset(split='train')
        val_dataset = EditingMixDataset(split='val')
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
        modelA = EK_VAE()
        modelB = PK_VAE()
        return modelA,modelB

    @staticmethod
    def get_optimizer(args,model):
        params = [p for p in model.parameters() if p.requires_grad]
        optimizer = optim.AdamW(params,
                              lr = args.lr,
                              weight_decay=args.weight_decay)
        return optimizer
    


    def train_one_epoch(self,model,optimizer,criterion,dataloader,device,epoch,args):
        """训练一个epoch"""
        model.train()  # set model to training mode
        for _,data in enumerate(tqdm(dataloader)):
            source_keypoint = data['source_keypoint'][:,:,1:2] # [b,26,1]
            target_keypoint = data['target_keypoint'][:,:,1:2] # [b,26,1]
            target_point = data['target_point'][:,:,1:2] # [b,257,1]
            source_param = data['source_param'].unsqueeze(-1) # [b,11,1]
            target_param = data['target_param'].unsqueeze(-1) # [b,11,1]
            source_keypoint = source_keypoint.to(device) 
            target_keypoint = target_keypoint.to(device) 
            target_point = target_point.to(device)
            source_param = source_param.to(device)
            target_param = target_param.to(device)
          
            optimizer.zero_grad()

            # # AE
            target_params_pred,target_point_pred = model(source_keypoint, target_keypoint, source_param)

            loss1 = criterion(target_params_pred,target_param)
            loss2 = criterion(target_point_pred,target_point)
            loss = loss1 + loss2
            # 反向传播
            loss.backward()

            # 参数更新
            optimizer.step()            
        # 打印loss
        print(f"train——epoch: {epoch}, loss: {loss.item()}, keypoint loss: {loss1.item()}, full loss: {loss2.item()}")  


    @torch.no_grad()
    def infer_keypoint(self,args, model, dataloader,device, epoch):
        """测试一个epoch"""
        model.eval()

        total_parsec_loss = [[0]*3 for _ in range(11)]
        
        correct_pred = 0  # 预测正确的样本数量
        total_pred = 0  # 总共的样本数量
        total_loss = 0.0

        test_loader = tqdm(dataloader)
        for _,data in enumerate(test_loader):
            
            gt = data['target_point'][:,:,1:2] # [b,257,1]
            x = data['target_point'][0,:,0] # [b, 257]
            source_keypoint = data['source_keypoint'][:,:,1:2] # [b,26,1]
            target_keypoint = data['target_keypoint'][:,:,1:2] # [b,26,1]
            source_param = data['source_param'].unsqueeze(-1) # [b,11,1]
            target_param = data['target_param'].unsqueeze(-1) # [b,11,1]
            source_keypoint = source_keypoint.to(device) 
            target_keypoint = target_keypoint.to(device) 
            source_param = source_param.to(device)
            target_param = target_param.to(device)
            gt = gt.to(device)
            _,recon_batch = model(source_keypoint, target_keypoint, source_param) 

            total_pred += source_keypoint.shape[0]

            loss = nn.MSELoss()(recon_batch[:,::10], target_keypoint)
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
        print(f"infer——epoch: {epoch}, avg_parsec_loss: {avg_parsec_loss_sci}")
        # 保存评测结果
        with open(f'{args.log_dir}/infer_editing_params_result.txt','a') as f:
            f.write(f"infer——epoch: {epoch}, accuracy: {accuracy}, keypoint_loss: {avg_loss:.2e}\n")
            f.write(f"infer——epoch: {epoch}, avg_parsec_loss: {' & '.join(avg_parsec_loss_sci)}\n")

 

    def main(self,args):
        """Run main training/evaluation pipeline."""
        
        # 单卡训练
        modelA,modelB = self.get_model(args)
        device = torch.device('cuda:2' if torch.cuda.is_available() else 'cpu')
        modelA.to(device)
        modelB.to(device)
        
        # # Check for a checkpoint editing model
        if len(args.checkpoint_path_A) > 0:
            assert os.path.isfile(args.checkpoint_path_A)
            load_checkpoint(args,args.checkpoint_path_A, modelA)
        
        # # Check for a checkpoint reconstruct model
        if len(args.checkpoint_path_B) > 0:
            assert os.path.isfile(args.checkpoint_path_B)
            load_checkpoint(args,args.checkpoint_path_B, modelB)

        model = EK_PKVAE(modelA,modelB)
        model.to(device)
        train_loader, val_loader = self.get_loaders(args) 
        optimizer = self.get_optimizer(args,model)
        criterion = nn.MSELoss()
        # criterion = chamfer_distance
        # Scheduler https://arxiv.org/pdf/1812.01187.pdf
        lf = lambda x: ((1 + math.cos(x * math.pi / args.max_epoch)) / 2) * (1 - args.lrf) + args.lrf  # cosine
        scheduler = lr_scheduler.LambdaLR(optimizer, lr_lambda=lf)
            
        for epoch in range(1,args.max_epoch+1):
            # train
            self.train_one_epoch(model=model,
                                 optimizer=optimizer,
                                 criterion=criterion,
                                 dataloader=train_loader,
                                 device=device,
                                 epoch=epoch,
                                 args=args
                                 )
            scheduler.step()
            # save model and validate
            if epoch % args.val_freq == 0 or epoch == 1:
                save_checkpoint(args, epoch, model, optimizer, scheduler)
                print("Validation begin.......")
                self.infer_keypoint(
                    args=args,
                    model=model,
                    dataloader=val_loader,
                    device=device, 
                    epoch=epoch, 
                    )
 

if __name__ == '__main__':
    opt = parse_option()
    torch.backends.cudnn.enabled = True
    # 启用cudnn的自动优化模式
    torch.backends.cudnn.benchmark = True
    # 设置cudnn的确定性模式，pytorch确保每次运行结果的输出都保持一致
    torch.backends.cudnn.deterministic = True

    trainer = Trainer()
    trainer.main(opt)