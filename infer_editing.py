import argparse
import os
import torch
import torch.nn as nn

from tqdm import tqdm 
from torch.utils.data import DataLoader

import torch.optim as optim 
import torch.optim.lr_scheduler as lr_scheduler
from models import AE_B_Attention,AE_A_variable
from datasets import AirFoilDataset2, EditingAirFoilDataset
import math 

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
    parser.add_argument('--checkpoint_path', default='eval_result/logs_edit_A/cond_ckpt_epoch_210.pth',help='Model checkpoint path')
    # parser.add_argument('--checkpoint_path', default='eval_result/logs_p/ckpt_epoch_120.pth',help='Model checkpoint path')
    parser.add_argument('--reconstruct_path', default='eval_result/logs_parsec_attention/ckpt_epoch_30000.pth',help='Model checkpoint path')
    parser.add_argument('--log_dir', default='test_result/logs_edit_A',
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

    checkpoint = torch.load(args.reconstruct_path, map_location='cpu')
    try:
        args.start_epoch = int(checkpoint['epoch']) + 1
    except Exception:
        args.start_epoch = 0
    model.load_state_dict(checkpoint['model'], strict=True)
    print("=> loaded successfully '{}' (epoch {})".format(
        args.checkpoint_path, checkpoint['epoch']
    ))

    del checkpoint
    torch.cuda.empty_cache()
    
def load_checkpoint_editing(args, model):
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

    del checkpoint
    torch.cuda.empty_cache()
count = 0 
loss_over = torch.zeros(1).to('cuda')
loss_all = [0] * 10
class Tester:
    @staticmethod
    def get_model(args):
        # model_A = AE_A()
        model_A = AE_A_variable()
        model = AE_B_Attention()
        return model, model_A
    
    def test(self,args):
        loss = nn.MSELoss(size_average=None, reduce=None, reduction='mean')
        import matplotlib.pyplot as plt
        """Run main training/testing pipeline."""
        test_dataset = EditingAirFoilDataset(split='test')
        test_loader = DataLoader(test_dataset,
                                 batch_size=1,
                                 shuffle=False,
                                 num_workers=args.num_workers)
        n_data_test = len(test_loader.dataset)
        test_loader = tqdm(test_loader)
        print(f"length of validating dataset: {n_data_test}")
        model, model_A = self.get_model(args)
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        load_checkpoint(args, model)
        load_checkpoint_editing(args, model_A)
        model.to(device)
        model_A.to(device)
        nm = os.path.basename(args.checkpoint_path).split('.')[0]
        os.makedirs(os.path.join(args.log_dir,nm), exist_ok=True)
        log_dir = os.path.join(args.log_dir,nm)
        model.eval()
        model_A.eval()
        for step, data in enumerate(test_loader):
            if step % 20 != 0:
                continue 
            if step > 1000:
                return 
            global count 
            count += 1
            fig, (ax1, ax2, ax3, ax4) = plt.subplots(1, 4, figsize=(20, 5))
            with torch.no_grad():
                x_sample = data['origin_input'] # [b,20,2]
                x_editing = data['editing_input'] # [b,20,2]
                x_physics = data['params'] # [b,10]
                
                x_physics = x_physics.unsqueeze(-1) #[b,10,1]
                
                # x_physics = x_physics.expand(-1,-1,2) #[b,9,2]
                
                y_physics = data['gt'].to(device)
                origin_data = data['origin_data']
                x_gt = data['origin_trans'].unsqueeze(-1) # [b,200,2]
                x_sample = x_sample.to(device) 
                x_editing = x_sample.to(device)             
                x_physics = x_physics.to(device)
                x_gt = x_gt.to(device)
                pred = model_A(x_sample, x_editing, x_physics)
                global loss_all
                global loss_over
                # print('------------------------------------------------')
                # print(pred)
                # print(y_physics)
                for ind in range(10):
                    # l = loss(pred[0,ind], y_physics[0,ind])
                    l = abs(pred[0,ind] - y_physics[0,ind]) 
                    loss_all[ind] += l
                ll = loss(pred, y_physics)
                loss_over += ll 
                # ----------------------------------------
                x_sample = data['editing_input'] # [b,20,2]
                x_physics = pred # [b,9]
                x_physics = x_physics.expand(-1,-1,2) #[b,10,2]
                
                y_physics = data['gt']
                y_physics = y_physics.unsqueeze(-1)
                y_physics = y_physics.expand(-1,-1,2) #[b,9,2]
                y_physics = y_physics.to(device)
                x_physics = x_physics.to(device)
                x_sample = x_sample.to(device)
                # # AE
                output = model(x_sample,x_physics) # [b,20,2],[b,10,2]
                output2 = model(x_sample,y_physics) # [b,20,2],[b,10,2]
                # data = x_gt
                # ---
                
                origin_x = origin_data[0,:,0].cpu().numpy()
                origin_y = origin_data[0,:,1].cpu().numpy()
                ax1.scatter(origin_x, origin_y, color='red', marker='o')
                ax1.set_xlabel('X')
                ax1.set_ylabel('Y')
                ax1.set_title('Source airfoil')
                
                ax2.set_title('Target airfoil')
                trans_x = x_gt[0,:,0].cpu().numpy()
                trans_y = x_gt[0,:,1].cpu().numpy()
                ax2.scatter(trans_x, trans_y, color='red', marker='o')
                ax2.set_xlabel('X')
                ax2.set_ylabel('Y')
                
                outputs2_x = output2[0,:,0].cpu().numpy()
                outputs2_y = output2[0,:,1].cpu().numpy()
                ax3.scatter(outputs2_x, outputs2_y, color='red', marker='o')
                ax3.set_xlabel('X')
                ax3.set_ylabel('Y')
                ax3.set_title('Target Gt Params')
                
                outputs_x = output[0,:,0].cpu().numpy()
                outputs_y = output[0,:,1].cpu().numpy()
                ax4.scatter(outputs_x, outputs_y, color='red', marker='o')
                ax4.set_xlabel('X')
                ax4.set_ylabel('Y')
                ax4.set_title('Target Predict Params')

                fig.tight_layout()

                plt.savefig(f'{log_dir}/{step}_editing.png', format='png')
                plt.close()
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
    ass = 0
    for a in loss_all:
        ass += a 
    print('loss ', [ll/count for ll in loss_all], '  all_mean ', ass / count, count)# loss_over.mean())# ass / count, count)
    
    
    """
    loss  [tensor([0.0082], device='cuda:0'), tensor([2.1007], device='cuda:0'), tensor([3.3069], device='cuda:0'), 
    tensor([9.3280], device='cuda:0'), tensor([3.3635], device='cuda:0'), tensor([2.5971], device='cuda:0'), 
    tensor([2.7121], device='cuda:0'), tensor([8.9056], device='cuda:0'), tensor([4.4089], device='cuda:0'), 
    tensor([0.0244], device='cuda:0')]   
    all_mean  tensor([36.7553], device='cuda:0')
    
    ----
    
    loss  [tensor([0.0072], device='cuda:0'), tensor([2.0569], device='cuda:0'), tensor([3.3067], device='cuda:0'), 
    tensor([9.3255], device='cuda:0'), tensor([3.3655], device='cuda:0'), tensor([2.6043], device='cuda:0'), 
    tensor([2.7219], device='cuda:0'), tensor([8.9115], device='cuda:0'), tensor([4.4052], device='cuda:0'), 
    tensor([0.0106], device='cuda:0')]   
    all_mean  tensor([36.7153], device='cuda:0'
    
    """