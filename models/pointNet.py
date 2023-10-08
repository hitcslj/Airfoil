import torch
from torch import nn
import torch.nn.functional as F


class pointNet(nn.Module):
  def __init__(self,input_channels=2):
    super(pointNet, self).__init__() 
    self.conv1 = nn.Conv1d(input_channels, 16, kernel_size=1, bias=False)
    self.conv2 = nn.Conv1d(16, 32, kernel_size=1, bias=False)
    self.conv3 = nn.Conv1d(32, 64, kernel_size=1, bias=False)
    self.conv4 = nn.Conv1d(64, 128, kernel_size=1, bias=False) 
    self.conv5 = nn.Conv1d(128, 256, kernel_size=1, bias=False)
    self.conv6 = nn.Conv1d(256, 512, kernel_size=1, bias=False)
    self.conv7 = nn.Conv1d(512, 1024, kernel_size=1, bias=False)

    self.conv8 = nn.Conv1d(1024*2, 512, kernel_size=1, bias=False)
    self.conv9 = nn.Conv1d(512, 128, kernel_size=1, bias=False)
    self.conv10 = nn.Conv1d(128, 64, kernel_size=1, bias=False)
    self.conv11 = nn.Conv1d(64, 16, kernel_size=1, bias=False)
    self.conv12 = nn.Conv1d(16, 2, kernel_size=1, bias=False)

    self.bn1 = nn.BatchNorm1d(16)
    self.bn2 = nn.BatchNorm1d(32)
    self.bn3 = nn.BatchNorm1d(64)
    self.bn4 = nn.BatchNorm1d(128)
    self.bn5 = nn.BatchNorm1d(256)
    self.bn6 = nn.BatchNorm1d(512)
    self.bn7 = nn.BatchNorm1d(1024)

    
    self.bn8 = nn.BatchNorm1d(512)
    self.bn9 = nn.BatchNorm1d(128)
    self.bn10 = nn.BatchNorm1d(64)
    self.bn11 = nn.BatchNorm1d(16)


    self.mid_conv = nn.Conv1d(1024, 1024, kernel_size=1, bias=False)
    self.mid_bn = nn.BatchNorm1d(1024)


  def forward(self, x):

    ## encoder
    x = F.leaky_relu(self.bn1(self.conv1(x))) # [B, 2, 200] -> [B, 16, 200]
    x = F.leaky_relu(self.bn2(self.conv2(x))) # [B, 16, 200] -> [B, 32, 200]
    x = F.leaky_relu(self.bn3(self.conv3(x))) # [B, 32, 200] -> [B, 64, 200]
    x = F.leaky_relu(self.bn4(self.conv4(x))) # [B, 64, 200] -> [B, 128, 200]
    x = F.leaky_relu(self.bn5(self.conv5(x))) # [B, 128, 200] -> [B, 256, 200]
    x = F.leaky_relu(self.bn6(self.conv6(x))) # [B, 256, 200] -> [B, 512, 200]
    point_feature = F.leaky_relu(self.bn7(self.conv7(x))) # [B, 512, 200] -> [B, 1024, 200]
    x = F.adaptive_max_pool1d(point_feature, 100)  # [B, 1024, 50]

    x = F.leaky_relu(self.mid_bn(self.mid_conv(x)))  # [B, 1024, 50]
    x = F.adaptive_avg_pool1d(x, 50)  # [B, 1024, 10]

    ## decoder
    x = x.repeat(1, 1, 4)  # [B, 1024, 200]
    x = torch.cat((x,point_feature),dim=1)  # [B, 1024*2, 200]
    x =  F.leaky_relu(self.bn8(self.conv8(x)))  # [B, 512, 200]
    x = F.leaky_relu(self.bn9(self.conv9(x)))  # [B, 128, 200]
    x = F.leaky_relu(self.bn10(self.conv10(x)))  # [B, 64, 200]
    x = F.leaky_relu(self.bn11(self.conv11(x)))  # [B, 16, 200]
    x = self.conv12(x)  # [B, 2, 200]
    return x



