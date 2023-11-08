import torch
from torch import nn
 
class AE_AB(nn.Module):
    def __init__(self,modelA,modelB) -> None:
        super().__init__()
        self.modelA = modelA
        self.modelB = modelB


    def forward(self,source_keypoint,target_keypoint,source_params): # y就是物理参数
        target_params_pred = self.modelA(source_keypoint,target_keypoint,source_params)
        target_params_pred_ = target_params_pred.expand(-1,-1,2)
        target_point_pred = self.modelB(target_keypoint,target_params_pred_) 
        return target_params_pred,target_point_pred
