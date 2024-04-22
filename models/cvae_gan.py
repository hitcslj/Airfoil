import torch
import torch.nn as nn
import torch.nn.functional as F


class Discriminator(nn.Module):
    def __init__(self, feature_size=257):
        super(Discriminator, self).__init__()
        self.feature_size = feature_size
        self.dis = nn.Sequential(
            nn.Linear(self.feature_size, 256),  # 输入特征数为784，输出为256
            nn.LeakyReLU(0.2),  # 进行非线性映射
            nn.Linear(256, 256),  # 进行一个线性映射
            nn.LeakyReLU(0.2),
            nn.Linear(256, 1),
            nn.Sigmoid()  # 也是一个激活函数，二分类问题中，
            # sigmoid可以班实数映射到【0,1】，作为概率值，
            # 多分类用softmax函数
        )

    def forward(self, x):
        x = x.squeeze(dim=-1)
        x = self.dis(x)
        return x

class CVAE_GAN(nn.Module):
    def __init__(self, feature_size=257, latent_size=10, condition_size=37):
        super(CVAE_GAN, self).__init__()
        self.feature_size = feature_size
        self.condition_size = condition_size
        self.latent_size = latent_size
 
        self.encoder_fc = nn.Sequential(
            nn.Linear(self.feature_size+self.condition_size, 256),
            nn.Tanh(),
            nn.Linear(256, 128),
            nn.Tanh(),
            nn.Linear(128, 64),
            nn.Tanh(),
            )

        self.fc_mean = nn.Linear(64, self.latent_size)
        self.fc_var = nn.Linear(64, self.latent_size)

   
        self.decoder_fc = nn.Sequential(
            nn.Linear(latent_size+condition_size, 64),
            nn.Tanh(),
            nn.Linear(64, 128),
            nn.Tanh(),
            nn.Linear(128, 256),
            nn.Tanh(),
            nn.Linear(256, self.feature_size),
            # nn.Tanh()
            )

        self.elu = nn.ELU()
        self.sigmoid = nn.Sigmoid()

    def encode(self, x, c): # Q(z|x, c)
        '''
        x: (bs, feature_size,1)
        c: (bs, class_size,1)
        '''
        inputs =  torch.cat([x,c],dim=1) # (bs,feature_size+condition_size,1)
        y = inputs.squeeze(dim=-1)
        h1 = self.encoder_fc(y)
        z_mu = self.fc_mean(h1)
        z_var = F.softplus(self.fc_var(h1))
        return z_mu, z_var

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5*logvar)
        eps = torch.randn_like(std)
        return mu + eps*std

    def decode(self, z, c): # P(x|z, c)
        '''
        z: (bs, latent_size)
        c: (bs, condition_size, 1)
        '''
        c = c.squeeze(dim=-1)
        y = torch.cat([z,c],dim=1)   
        return self.decoder_fc(y)
    
    def sample(self,c):
        '''
        z: (bs, latent_size)
        c: (bs, condition_size, 1)
        '''
        batch = c.shape[0]
        z = torch.randn((batch,self.latent_size)).to(c.device)
        recons_batch = self.decode(z, c).reshape(-1,self.feature_size,1)
        return recons_batch
    
    def forward(self, x, c):
        # x: (bs, feature_size, 1)
        # c: (bs, condition_size, 1)
        mu, logvar = self.encode(x, c)
        z = self.reparameterize(mu, logvar)
        recons_batch = self.decode(z, c).reshape(-1,self.feature_size,1)
        return recons_batch, mu, logvar
  
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