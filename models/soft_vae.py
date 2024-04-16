# -*- coding: utf-8 -*-
"""
Created on Tue Nov  9 16:11:30 2021

@author: XieHR
"""


import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data as Data
import argparse
import time

from hyperspherical_vae.distributions import VonMisesFisher
from hyperspherical_vae.distributions import HypersphericalUniform

import matplotlib.pyplot as plt
plt.style.use('seaborn-paper')


# 设置模型运行的设备
cuda = torch.cuda.is_available()
device = torch.device("cuda" if cuda else "cpu")

# 设置默认参数
parser = argparse.ArgumentParser(description="Variational Auto-Encoder")
# parser.add_argument('--result_dir', type=str, default='./VAEResult', metavar='DIR', help='output directory')
parser.add_argument('--z_dim', type=int, default=10, metavar='N', help='the dim of latent variable z(default: 20)')
parser.add_argument('--batch_size', type=int, default=16, metavar='N', help='batch size for training(default: 128)')
parser.add_argument('--lr', type=float, default=5.0e-4, help='learning rate(default: 0.001)')
parser.add_argument('--epochs', type=int, default=5000, metavar='N', help='number of epochs to train(default: 200)')
parser.add_argument('--alpha1', type=float, default=1.0e5, help='mse_loss * alpha1 (default: 5.0e4)')
parser.add_argument('--alpha2', type=float, default=1.0e-1, help='kld_loss * alpha2 (default: 1.0e-1)')
parser.add_argument('--mode', type=int, default=0, help='N-VAE : 0, S-VAE: 1(default: 0)')
parser.add_argument('--resume', type=bool, default=False, help='resume: True or False(default: False)')
args = parser.parse_args()

# nx=200   # airfoil y coordinates
# ny=3     # number of conditions/labels
# nz=10   # latent vector 
# batch_size=16
# num_epoch=5000

alpha1,alpha2=5.0e4,0.1


class VAE(nn.Module):
    def __init__(self,x_dim,z_dim,distribution='normal'):
        super(VAE, self).__init__()
        self.x_dim, self.z_dim, self.distribution = x_dim,z_dim,distribution
        
        self.encoder_fc = nn.Sequential(
            nn.Linear(self.x_dim, 256),
            nn.Tanh(),
            nn.Linear(256, 128),
            nn.Tanh(),
            nn.Linear(128, 64),
            nn.Tanh(),
            )

        self.fc_mean = nn.Linear(64, self.z_dim)
        if self.distribution == 'normal':
            self.fc_var = nn.Linear(64, self.z_dim)
        elif self.distribution == 'vmf':
            self.fc_var = nn.Linear(64, 1)
   
        self.decoder_fc = nn.Sequential(
            nn.Linear(z_dim, 64),
            nn.Tanh(),
            nn.Linear(64, 128),
            nn.Tanh(),
            nn.Linear(128, 256),
            nn.Tanh(),
            nn.Linear(256, x_dim),
            # nn.Tanh()
            )
        
    def encoder(self, x):
        x = self.encoder_fc(x)
        
        if self.distribution == 'normal':
            # compute mean and std of the normal distribution
            z_mean = self.fc_mean(x)
            z_var = F.softplus(self.fc_var(x))
        elif self.distribution == 'vmf':
            # compute mean and concentration of the von Mises-Fisher
            z_mean = self.fc_mean(x)
            z_mean = z_mean / z_mean.norm(dim=-1, keepdim=True)
            # the `+ 1` prevent collapsing behaviors
            z_var = F.softplus(self.fc_var(x)) + 1
        
        return z_mean, z_var

    
    def reparameterize(self, z_mean, z_var):
        if self.distribution == 'normal':
            q_z = torch.distributions.normal.Normal(z_mean, z_var)
            p_z = torch.distributions.normal.Normal(torch.zeros_like(z_mean), torch.ones_like(z_var))
        elif self.distribution == 'vmf':
            q_z = VonMisesFisher(z_mean, z_var)
            p_z = HypersphericalUniform(self.z_dim - 1)
        else:
            raise NotImplementedError

        return q_z, p_z
    
    def decoder(self,z):
        x = self.decoder_fc(z)
        return x
    
    def forward(self,x):
        z_mean, z_var = self.encoder(x)
        q_z, p_z = self.reparameterize(z_mean, z_var)
        z = q_z.rsample()
        x_ = self.decoder(z)
        return (z_mean, z_var), (q_z, p_z), z, x_
    
    def initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight.data)
                # nn.init.normal_(m.weight.data, 0, 0.02)
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            else:
                nn.init.xavier_normal_(m.weight.data,validate_args=False)
                m.bias.data.zero_(validate_args=False)

def log_likelihood(model, x, n=10):
    """
    :param model: model object
    :param optimizer: optimizer object
    :param n: number of MC samples
    :return: MC estimate of log-likelihood
    """

    z_mean, z_var = model.encode(x.reshape(-1, 784))
    q_z, p_z = model.reparameterize(z_mean, z_var)
    z = q_z.rsample(torch.Size([n]))
    x_mb_ = model.decode(z)

    log_p_z = p_z.log_prob(z)

    if model.distribution == 'normal':
        log_p_z = log_p_z.sum(-1)

    log_p_x_z = -nn.BCEWithLogitsLoss(reduction='none')(x_mb_, x.reshape(-1, 784).repeat((n, 1, 1))).sum(-1)

    log_q_z_x = q_z.log_prob(z)

    if model.distribution == 'normal':
        log_q_z_x = log_q_z_x.sum(-1)

    return ((log_p_x_z + log_p_z - log_q_z_x).t().logsumexp(-1) - np.log(n)).mean()


def lossFunction(recon_x,x,mu,logstd):
    # mu 
    # logstd is log(sigma**2)
    
    # 1. the reconstruction loss.
    MSE = F.mse_loss(recon_x, x)
    # 2. KL-divergence
    KLD = 0.5 * torch.sum(torch.exp(logstd) + torch.pow(mu, 2) - 1. - logstd)/x.size(0)

    L1 = F.l1_loss(recon_x, x,reduction="none")
    
    loss = args.alpha1*MSE+args.alpha2*KLD 
    return loss,MSE,KLD,L1

def loadData():
    airfoil_coords = np.load('../airfoils.npz')
    xs = airfoil_coords['xs']     # x coordinates
    ys = airfoil_coords['ys']  # airfoil y coordinates 
    namelist = airfoil_coords['filelist']
    
    # remove Le point
    xs = np.delete(xs,[100])
    ys = np.delete(ys,[100],axis=1)
    # remove some bad airfoils
    # ys = np.delete(ys,[1086],axis=0)
    
    # make tensor
    xs = torch.tensor(xs).float()
    ys = torch.tensor(ys).float()
    
    x = ys
    y = cal_y(xs, ys)
    
    return xs,x,y,namelist

def cal_y(xs,ys):
    """
    >>> calculate labels/Conditions
    # input airfoil x, y coordinates
    # output : max_thick position
    #          max thickness
    #          te thickness
    #          max_camber position
    #          max_camber
    #          in N*5 shape tensor
    """
    iLE = 100 # leading edge point (0,0)
    x = xs[iLE:]
    us = ys[:,:iLE].flip(1)
    ls = ys[:,iLE:]
    
    thickness = us - ls
    max_thick_y, imax_thick = torch.max(thickness,dim=1)
    max_thick_x = x[imax_thick]
    
    te_thick = us[:,-1] - ls[:,-1]
    
    camber = (us+ls)/2.0
    max_camber_y,imax_camber = torch.max(camber,dim=1)
    
    max_camber_x = x[imax_camber]
    max_camber_x[max_camber_y == 0.]=0.5
    # if max_camber_y == 0.:
    #     max_camber_x = 0.5
    # else:
    #     max_camber_x = x[imax_camber]
    
    return torch.vstack((max_thick_x,max_thick_y,te_thick,
                         max_camber_x,max_camber_y)).T
    
    
def normalization(x, y):
    # norm x
    Min = torch.min(x)
    Max = torch.max(x)

    offsetx = Min
    scalex = 2.0/(Max-Min)

    x = (x-offsetx)*scalex-1.0
    print('{:=^60}'.format(" normalization x "))
    print("x Min = ",Min.data)
    print("x Max = ",Max.data)
    
    # norm y
    Min = torch.min(y, 0)[0]
    Max = torch.max(y, 0)[0]

    offsety = Min
    scaley = 2.0/(Max-Min)

    y = (y-offsety)*scaley-1.0
    
    dataNorm = scalex,offsetx,scaley,offsety
    print('{:=^60}'.format(" normalization y "))
    print("max_thick_x,max_thick_y,te_thick")
    print("y Min = ",Min.data)
    print("y Max = ",Max.data)
    return x, y, dataNorm

def deNorm(data,dataNorm,mode='x'):
    scalex,offsetx,scaley,offsety =dataNorm
    if mode=='x':
        data=(data+1.0)/scalex+offsetx
    elif mode=='y':
        data=(data+1.0)/scaley+offsety
        
    return data


def mk_dataloader():
    xs,x,y,namelist = loadData()
    
    x,y,dataNorm = normalization(x, y)

    dataset = Data.TensorDataset(x)
    
    n1 = int(x.shape[0] *1.0)
    n2 = x.shape[0] - n1
    
    training_set,test_set = Data.random_split(dataset,[n1,n2],
                                              generator=torch.Generator().manual_seed(42))
    # print(training_set.shape)
    
    return training_set,test_set,dataNorm
   

        
def train(model, optimizer, dataloader, resume=False):
    
    casename = "VAE_"+model.distribution+"_z"+str(model.z_dim)
    f = open(casename+'.log','w')
    # print(casename)
    if resume:
        state = torch.load(casename+'.pth')
        model.load_state_dict(state['net'])
        optimizer.load_state_dict(state['optimizer'])
        start_epoch = state['epoch']
    else:
        start_epoch = 0
        
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 
                                                step_size=2500, 
                                                gamma=0.1)
    # scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=200, eta_min=1e-6)
    
    
    print('{:=^60}'.format(" start training "))
    for epoch in range(start_epoch, args.epochs):
        loss_MSE = 0.
        loss_KLD = 0.
        loss_MAE = 0.
        LOSS = 0.
        for i, (x,) in enumerate(dataloader):
            x = x.to(device)
            n_local = x.shape[0]
            optimizer.zero_grad()
            
            _, (q_z, p_z), _, x_ = model(x)

            loss_mse = F.mse_loss(x, x_)

            if model.distribution == 'normal':
                loss_kl = torch.distributions.kl.kl_divergence(q_z, p_z).sum(-1).mean()
            elif model.distribution == 'vmf':
                loss_kl = torch.distributions.kl.kl_divergence(q_z, p_z).mean()
            else:
                raise NotImplementedError

            loss = args.alpha1*loss_mse + args.alpha2*loss_kl
            L1 = F.l1_loss(x_, x,reduction="none")
            
            loss.backward()
            optimizer.step()       
            
            loss_mae = torch.mean(L1)
            
            loss_MSE += loss_mse*n_local/nt
            loss_KLD += loss_kl*n_local/nt
            loss_MAE += loss_mae*n_local/nt
            LOSS += loss*n_local/nt
                
        if (epoch + 1) % 10 == 0:
            info = "epoch[{}/{}],mse:{:.6f},kld:{:.6f},vae:{:.6f},mae:{:.6f}".format(
                      epoch+1, args.epochs,
                      loss_MSE, loss_KLD, LOSS, loss_MAE)
            print(info)
            print(info,file=f)
            # print(optimizerN.state_dict)

    
        scheduler.step()
    
    state = {'net':model.state_dict(), 'optimizer':optimizer.state_dict(), 'epoch':epoch}
    torch.save(state, casename+'.pth')
    f.close()
    
if __name__ == "__main__":
    
    training_set,test_set,dataNorm = mk_dataloader()
    nt = len(training_set)
    nx = training_set[0][0].shape[0]
    nz = args.z_dim

    dataloader = Data.DataLoader(training_set, 
                                 batch_size=args.batch_size, 
                                 shuffle=True)
    distribution=['normal','vmf']

    modelN = VAE(nx,nz, distribution=distribution[args.mode]).to(device)
    print(modelN)
    optimizerN = torch.optim.Adam(modelN.parameters(), lr=args.lr)
    train(modelN, optimizerN, dataloader,resume=args.resume)
    
