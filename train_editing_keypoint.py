import os
import math 
import argparse
import torch
from torch import nn
from torch.utils.data import DataLoader
import torch.optim as optim 
import torch.optim.lr_scheduler as lr_scheduler
from tqdm import tqdm
from models import AE_A_Keypoint
from dataload import EditingMixDataset
    
def parse_option():
    """Parse cmd arguments."""
    parser = argparse.ArgumentParser()
    # Data
    parser.add_argument('--batch_size', type=int, default=512,
                        help='Batch Size during training')
    parser.add_argument('--num_workers',type=int,default=8)
    # Training
    parser.add_argument('--start_epoch', type=int, default=1)
    parser.add_argument('--max_epoch', type=int, default=1001)
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
    parser.add_argument('--checkpoint_path', default='',help='Model checkpoint path')  
    parser.add_argument('--log_dir', default='weights/logs_edit_keypoint',
                        help='Dump dir to save model checkpoint')
    parser.add_argument('--val_freq', type=int, default=1000)  # epoch-wise
    parser.add_argument('--save_freq', type=int, default=1)  # epoch-wise
    

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
    def get_model():
        model = AE_A_Keypoint()
        return model

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
        train_loss = 0
        total_pred = 0  # 总共的样本数量
        for _,data in enumerate(tqdm(dataloader)):
            source_keypoint = data['source_keypoint'][:,:,1:2] # [b,26,1]
            target_keypoint = data['target_keypoint'][:,:,1:2] # [b,26,1]
            source_param = data['source_param'].unsqueeze(-1) # [b,11,1]
            target_param = data['target_param'].unsqueeze(-1) # [b,11,1]
            source_keypoint = source_keypoint.to(device) 
            target_keypoint = target_keypoint.to(device) 
            source_param = source_param.to(device)
            target_param = target_param.to(device)

            optimizer.zero_grad()

            # # AE
            pred = model(source_keypoint, target_keypoint, source_param) 
          
            loss = criterion(target_param,pred)
            train_loss += loss.item()
            total_pred += source_keypoint.shape[0]
            # 反向传播
            loss.backward()

            # 参数更新
            optimizer.step()            
        # 打印loss
        print(f"train——epoch: {epoch}, loss: {train_loss/total_pred}")  


    @torch.no_grad()
    def evaluate_one_epoch(self, model,criterion, dataloader,device, epoch, args):
        """验证一个epoch"""
        model.eval()
        
        total_pred = 0  # 总共的样本数量
        total_loss = 0.0

        test_loader = tqdm(dataloader)
        for _,data in enumerate(test_loader):
             
            source_keypoint = data['source_keypoint'][:,:,1:2] # [b,26,1]
            target_keypoint = data['target_keypoint'][:,:,1:2] # [b,26,1]
            source_param = data['source_param'].unsqueeze(-1) # [b,11,1]
            target_param = data['target_param'].unsqueeze(-1) # [b,11,1]
            source_keypoint = source_keypoint.to(device) 
            target_keypoint = target_keypoint.to(device) 
            source_param = source_param.to(device)
            target_param = target_param.to(device)


            total_pred += source_keypoint.shape[0]
            # loss,_ = criterion(data,output)
            pred = model(source_keypoint, target_keypoint, source_param) 
          
            loss = criterion(target_param,pred)
            total_loss += loss.item()
        # accuracy = correct_pred / total_pred
        avg_loss = total_loss / total_pred
        
        print(f"eval——epoch: {epoch}, avg_loss: {avg_loss}")
 

    def main(self,args):
        """Run main training/evaluation pipeline."""
        
        # 单卡训练
        model = self.get_model()
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model.to(device)

        train_loader, val_loader = self.get_loaders(args) 
        optimizer = self.get_optimizer(args,model)
        criterion = nn.MSELoss()
 
        # Scheduler https://arxiv.org/pdf/1812.01187.pdf
        lf = lambda x: ((1 + math.cos(x * math.pi / args.max_epoch)) / 2) * (1 - args.lrf) + args.lrf  # cosine
        scheduler = lr_scheduler.LambdaLR(optimizer, lr_lambda=lf)
      
 
        # # Check for a checkpoint
        if len(args.checkpoint_path)>0:
            assert os.path.isfile(args.checkpoint_path)
            load_checkpoint(args, model, optimizer, scheduler)
            
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
            args.val_freq = 1
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