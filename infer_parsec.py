import argparse
import os
import torch
import torch.nn as nn

from tqdm import tqdm 
from torch.utils.data import DataLoader

from models import AE_B
from datasets import AirFoilDataset2


def parse_option():
    """Parse cmd arguments."""
    parser = argparse.ArgumentParser()
    # Data
    parser.add_argument('--dataset', type=str, default=['airfoil'],# ['cocodatasets'],
                        nargs='+', help='list of datasets to train on')
    parser.add_argument('--test_dataset', default='airfoil')# 'cocodatasets')
    parser.add_argument('--data_root', default='./data/airfoil/picked_uiuc/')
    parser.add_argument('--num_workers',type=int,default=4)

    # io
    parser.add_argument('--checkpoint_path', default='eval_result/logs_parsec/ckpt_epoch_30000.pth',help='Model checkpoint path')
    parser.add_argument('--log_dir', default='./test_result/logs_parsec',
                        help='Dump dir to save visual result')

    parser.add_argument('--eval', default=False, action='store_true')

    # 评测指标相关
    parser.add_argument('--distance_threshold', type=float, default=0.01) # xy点的距离小于该值，被认为是预测正确的点
    parser.add_argument('--threshold_ratio', type=float, default=0.75) # 200个点中，预测正确的点超过一定的比例，被认为是预测正确的样本
   
    args, _ = parser.parse_known_args()


    return args

# BRIEF load checkpoint.
def load_checkpoint(args, model):
    """Load from checkpoint."""
    print("=> loading checkpoint '{}'".format(args.checkpoint_path))

    checkpoint = torch.load(args.checkpoint_path, map_location='cpu')
    try:
        args.start_epoch = int(checkpoint['epoch']) + 1
    except Exception:
        args.start_epoch = 0
    model.load_state_dict(checkpoint['model'], strict=True)
    print("=> loaded successfully '{}' (epoch {})".format(
        args.checkpoint_path, checkpoint['epoch']
    ))
    epoch = checkpoint['epoch']

    del checkpoint
    torch.cuda.empty_cache()

    return epoch
class Tester:

    @staticmethod
    def get_model(args):
        model = AE_B()
 
        return model
    
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
            pred = model(x_sample,x_physics) # [b,20,2],[b,9,2]

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
        
        s = f"eval——epoch: {epoch}, accuracy: {accuracy}, avg_loss: {avg_loss}"
        print(s)
        with open(os.path.join(args.log_dir,'eval_result.txt'),'a') as f:
            f.write(s+'\n')

    def test(self,args):
        import matplotlib.pyplot as plt
        """Run main training/testing pipeline."""
        test_dataset = AirFoilDataset2(split='test')
        test_loader = DataLoader(test_dataset,
                                 batch_size=1,
                                 shuffle=False,
                                 num_workers=args.num_workers)
        n_data_test = len(test_loader.dataset)
        test_loader = tqdm(test_loader)
        print(f"length of validating dataset: {n_data_test}")
        model = self.get_model(args)
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model.to(device)
        epoch = load_checkpoint(args, model)


        os.makedirs(args.log_dir, exist_ok=True)
        model.eval()
        for step, data in enumerate(test_loader):
            if step % 20 != 0: continue
            fig, (ax3, ax1, ax2) = plt.subplots(1, 3, figsize=(10, 5))
            with torch.no_grad():
                x_sample = data['input'] # [b,20,2]
                x_physics = data['params'] # [b,9]
                x_physics = x_physics.unsqueeze(-1) #[b,9,1]
                x_physics = x_physics.expand(-1,-1,2) #[b,9,2]

                x_gt = data['output'] # [b,200,2]
                x_sample = x_sample.to(device) 
                x_physics = x_physics.to(device)
                x_gt = x_gt.to(device)
                # # AE
                output = model(x_sample,x_physics) # [b,200,2],[b,9,2]
                data = x_gt
                origin_x = data[0,:,0].cpu().numpy()
                origin_y = data[0,:,1].cpu().numpy()
                ax1.scatter(origin_x, origin_y, color='red', marker='o')
                ax1.set_xlabel('X')
                ax1.set_ylabel('Y')
                # ax1.set_aspect('equal')
                # ax1.axis('off')
                ax1.set_title('Original Data')
                
                outputs_x = output[0,:,0].cpu().numpy()
                outputs_y = output[0,:,1].cpu().numpy()
                ax2.scatter(outputs_x, outputs_y, color='red', marker='o')
                ax2.set_xlabel('X')
                ax2.set_ylabel('Y')
                # ax2.set_aspect('equal')
                # ax2.axis('off')
                ax2.set_title('Predicted Data')

                sample_x = x_sample[0,:,0].cpu().numpy()
                sample_y = x_sample[0,:,1].cpu().numpy()
                ax3.scatter(sample_x, sample_y, color='red', marker='o')
                ax3.set_xlabel('X')
                ax3.set_ylabel('Y')
                ax3.set_title('Sample keypoints')

                fig.tight_layout()


                plt.savefig(f'{args.log_dir}/{step}_mlp.png', format='png')
                plt.close()
        self.evaluate_one_epoch(model,nn.MSELoss(),test_loader,device,epoch,args)
if __name__ == '__main__':
    opt = parse_option()
    # cudnn
    # 启用cudnn
    torch.backends.cudnn.enabled = True
    # 启用cudnn的自动优化模式
    torch.backends.cudnn.benchmark = True
    # 设置cudnn的确定性模式，pytorch确保每次运行结果的输出都保持一致
    torch.backends.cudnn.deterministic = True

    tester = Tester()
    tester.test(opt)