import argparse
import os
import torch
import torch.nn as nn
from tqdm import tqdm 
from torch.utils.data import DataLoader
import torch.optim as optim 
 
import torch.optim.lr_scheduler as lr_scheduler
from models import pointNet,AEBackboneNet,FoldingNet,AE_B
from datasets import AirFoilDataset2 
import math 
from pytorch3d.loss import chamfer_distance
    
def parse_option():
    """Parse cmd arguments."""
    parser = argparse.ArgumentParser()
    # Data
    parser.add_argument('--batch_size', type=int, default=64,
                        help='Batch Size during training')
    parser.add_argument('--num_workers',type=int,default=4)
    # Training
    parser.add_argument('--start_epoch', type=int, default=1)
    parser.add_argument('--max_epoch', type=int, default=30000)
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
    parser.add_argument('--checkpoint_path', default='./eval_result/logs_p/ckpt_epoch_last.pth',help='Model checkpoint path')
    parser.add_argument('--log_dir', default='./eval_result/logs_p',
                        help='Dump dir to save model checkpoint')
    parser.add_argument('--val_freq', type=int, default=100)  # epoch-wise
    parser.add_argument('--save_freq', type=int, default=10000)  # epoch-wise
    

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
    # optimizer.load_state_dict(checkpoint['optimizer'])
    # scheduler.load_state_dict(checkpoint['scheduler'])

    print("=> loaded successfully '{}' (epoch {})".format(
        args.checkpoint_path, checkpoint['epoch']
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
        train_dataset = AirFoilDataset2(split='train')
        val_dataset = AirFoilDataset2(split='val')
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
                                  shuffle=True,
                                  batch_size=args.batch_size,
                                  num_workers=args.num_workers)
        return train_loader,val_loader

    @staticmethod
    def get_model(args):
        # model = pointNet(input_channels=2)
        # model = AEBackboneNet(in_channels=2)
        # model = FoldingNet()
        model = AE_B()
        return model

    @staticmethod
    def get_optimizer(args,model):
        params = [p for p in model.parameters() if p.requires_grad]
        # optimizer = optim.SGD(params,
        #                       lr = args.lr*0.1,
        #                       momentum = 0.9,
        #                       weight_decay = 5E-5)
        optimizer = optim.AdamW(params,
                              lr = args.lr,
                              weight_decay=args.weight_decay)
        return optimizer
    


    def train_one_epoch(self,model,optimizer,criterion,dataloader,device,epoch,args):
        """训练一个epoch"""
        model.train()  # set model to training mode
        # L1 = nn.L1Loss()
        # L2 = nn.MSELoss()
        for _,data in enumerate(tqdm(dataloader)):
            x_sample = data['input'] # [b,20,2]
            x_physics = data['params'] # [b,9]
            x_physics = x_physics.unsqueeze(-1) #[b,9,1]
            x_physics = x_physics.expand(-1,-1,2) #[b,9,2]

            x_gt = data['output'] # [b,200,2]
            x_sample = x_sample.to(device) 
            x_physics = x_physics.to(device)
            x_gt = x_gt.to(device)
            optimizer.zero_grad()
            
            # # PointNet/FoldingNet
            # pred = model(data.permute(0,2,1)) # [b,2,200]
            # pred = pred.permute(0,2,1) # [b,200,2]

            # # AE
            pred = model(x_sample,x_physics) # [b,200,2],[b,9,2]
            
            # loss,_ = criterion(data,pred)
            loss = criterion(x_gt,pred)
            # 反向传播
            loss.backward()

            # 参数更新
            optimizer.step()            
        # 打印loss
        print(f"train——epoch: {epoch}, loss: {loss.item()}")  


    @torch.no_grad()
    def evaluate_one_epoch(self, model,criterion, dataloader,device, epoch, args):
        """验证一个epoch"""
        model.eval()
        
        correct_pred = 0  # 预测正确的样本数量
        total_pred = 0  # 总共的样本数量
        total_loss = 0.0

        test_loader = tqdm(dataloader)
        for _,data in enumerate(test_loader):
             
            x_sample = data['input'] # [b,20,2]
            x_physics = data['params'] # [b,9]
            x_physics = x_physics.unsqueeze(-1) #[b,9,1]
            x_physics = x_physics.expand(-1,-1,2) #[b,9,2]

            x_gt = data['output'] # [b,200,2]
            x_sample = x_sample.to(device) 
            x_physics = x_physics.to(device)
            x_gt = x_gt.to(device)
            pred = model(x_sample,x_physics) # [b,200,2],[b,9,2]

            total_pred += x_sample.shape[0]
            # loss,_ = criterion(data,output)
            loss = criterion(x_gt,pred)
            total_loss += loss.item()
            # 判断样本是否预测正确
            distances = torch.norm(x_gt - pred,dim=2) #(B,200)
            # 点的直线距离小于t，说明预测值和真实值比较接近，认为该预测值预测正确
            t = args.distance_threshold
            # 200个点中，预测正确的点的比例超过ratio，认为该形状预测正确
            ratio = args.threshold_ratio
            count = (distances < t).sum(1) #(B) 一个样本中预测坐标和真实坐标距离小于t的点的个数
            correct_count = (count >= ratio*200).sum().item() # batch_size数量的样本中，正确预测样本的个数
            correct_pred += correct_count
            
        accuracy = correct_pred / total_pred
        avg_loss = total_loss / total_pred
        
        print(f"eval——epoch: {epoch}, accuracy: {accuracy}, avg_loss: {avg_loss}")
 

    def main(self,args):
        """Run main training/evaluation pipeline."""
        
        # 单卡训练
        model = self.get_model(args)
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model.to(device)

        train_loader, val_loader = self.get_loaders(args) 
        optimizer = self.get_optimizer(args,model)
        criterion = nn.L1Loss()
        # criterion = chamfer_distance
        # Scheduler https://arxiv.org/pdf/1812.01187.pdf
        lf = lambda x: ((1 + math.cos(x * math.pi / args.max_epoch)) / 2) * (1 - args.lrf) + args.lrf  # cosine
        scheduler = lr_scheduler.LambdaLR(optimizer, lr_lambda=lf)
      
 
        # # Check for a checkpoint
        # if args.checkpoint_path:
        #     assert os.path.isfile(args.checkpoint_path)
        #     load_checkpoint(args, model, optimizer, scheduler)
            
        for epoch in range(args.start_epoch,args.max_epoch+1):
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
            if epoch % args.val_freq == 0:
                save_checkpoint(args, epoch, model, optimizer, scheduler)
                print("Validation begin.......")
                self.evaluate_one_epoch(
                    model=model,
                    criterion=criterion, 
                    dataloader=val_loader,
                    device=device, 
                    epoch=epoch, 
                    args=args)
 

if __name__ == '__main__':
    opt = parse_option()
    torch.backends.cudnn.enabled = True
    # 启用cudnn的自动优化模式
    torch.backends.cudnn.benchmark = True
    # 设置cudnn的确定性模式，pytorch确保每次运行结果的输出都保持一致
    torch.backends.cudnn.deterministic = True

    trainer = Trainer()
    trainer.main(opt)