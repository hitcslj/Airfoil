import torch
from torch import nn
import math


# class CVAE_Y(nn.Module):
#     def __init__(self, feature_size=257, latent_size=20, condition_size=37):
#         super(CVAE_Y, self).__init__()
#         self.feature_size = feature_size
#         self.condition_size = condition_size
#         self.latent_size = latent_size

#         # encode
#         self.fc1  = nn.Linear(feature_size + condition_size, 512)
#         self.fc21 = nn.Linear(512, latent_size)
#         self.fc22 = nn.Linear(512, latent_size)

#         # decode
#         self.fc3 = nn.Linear(latent_size + condition_size, 512)
#         self.fc4 = nn.Linear(512, feature_size)

#         self.elu = nn.ELU()
#         self.sigmoid = nn.Sigmoid()
 
#     def encode(self, x, c): # Q(z|x, c)
#         '''
#         x: (bs, feature_size)
#         c: (bs, class_size)
#         '''
#         inputs = torch.cat([x, c], 1) # (bs, feature_size+condition_size)
#         h1 = self.elu(self.fc1(inputs))
#         z_mu = self.fc21(h1)
#         z_var = self.fc22(h1)
#         return z_mu, z_var

#     def reparameterize(self, mu, logvar):
#         std = torch.exp(0.5*logvar)
#         eps = torch.randn_like(std)
#         return mu + eps*std

#     def decode(self, z, c): # P(x|z, c)
#         '''
#         z: (bs, latent_size)
#         c: (bs, condition_size)
#         '''
#         inputs = torch.cat([z, c], 1) # (bs, latent_size+condition_size)
#         h3 = self.elu(self.fc3(inputs))
#         return self.fc4(h3)

#     def forward(self, x, c):
#         mu, logvar = self.encode(x.reshape(-1, self.feature_size), c.reshape(-1,self.condition_size))
#         z = self.reparameterize(mu, logvar)
#         recons = self.decode(z, c.reshape(-1,self.condition_size))
#         # 对recons 要求和c保持一致
#         # vaild = self.discriminator(recons, c)

#         return recons, mu, logvar

#     def sample(self,c):
#         batch = c.shape[0]
#         z = torch.randn((batch,self.latent_size)).to(c.device)
#         recons_batch = self.decode(z, c.reshape(-1,self.condition_size))
#         return recons_batch.reshape(-1,257,2)



class CVAE_Y(nn.Module):
    def __init__(self, feature_size=257, latent_size=20, condition_size=37,attention_type='self'):
        super(CVAE_Y, self).__init__()
        self.feature_size = feature_size
        self.condition_size = condition_size
        self.latent_size = latent_size
        self.attention_type = attention_type

        # encode
        self.qkv = nn.Linear(1, 12)
        self.cq = nn.Linear(1, 4)
        self.cv = nn.Linear(1, 4)
        self.ck = nn.Linear(1, 4)
        self.fc0 = nn.Linear(feature_size, feature_size+condition_size)
        self.fc1  = nn.Linear(feature_size + condition_size, 512)
        
        self.fc21 = nn.Linear(512, latent_size)
        self.fc22 = nn.Linear(512, latent_size)

        # decode
        self.fc3 = nn.Linear(latent_size + condition_size, 512)
        self.fc4 = nn.Linear(512, feature_size)

        self.elu = nn.ELU()
        self.sigmoid = nn.Sigmoid()

    def attention(self, x, c, attention_type = 'self'):
        # input: x: (bs, feature_size, 1)
        #         c: (bs, condition_size, 1)
        # output: y: (bs, feature_size+condition_size)
        if attention_type == 'self':
            inputs = torch.cat([x, c], 1) # (bs, feature_size+condition_size,1)
            # 进行self-attention
            q,k,v = self.qkv(inputs).chunk(3, dim = -1)
            # Compute attention scores
            scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(k.size(-1))
            # Apply softmax to get attention distribution
            attn = torch.nn.functional.softmax(scores, dim=-1)
            # Compute weighted sum of v
            y = torch.matmul(attn, v) # (bs, feature_size+condition_size,4)
            y = y.sum(dim = -1) # (bs, feature_size+condition_size)
            return y

        elif attention_type == 'cross':
            # 进行cross-attention
            q = self.cq(x) # (bs, feature_size, 4)
            k, v = self.ck(c), self.cv(c) # (bs, condition_size, 4)
            # Compute attention scores
            scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(k.size(-1))
            # Apply softmax to get attention distribution
            attn = torch.nn.functional.softmax(scores, dim=-1)
            # Compute weighted sum of v
            y = torch.matmul(attn, v) # (bs, feature_size,4)
            y = y.sum(dim = -1) # (bs, feature_size)
            y = self.fc0(y)
            return y

    def encode(self, x, c, attention_type = 'self'): # Q(z|x, c)
        '''
        x: (bs, feature_size)
        c: (bs, class_size)
        '''
        # inputs进行self-attention
        inputs = self.attention(x,c,attention_type = attention_type) # (bs, feature_size+condition_size)
        h1 = self.elu(self.fc1(inputs))
        z_mu = self.fc21(h1)
        z_var = self.fc22(h1)
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
        c = c.reshape(-1,self.condition_size)
        inputs = torch.cat([z, c], 1) # (bs, latent_size+condition_size)
        h3 = self.elu(self.fc3(inputs))
        return self.fc4(h3)
    
    def sample(self,c):
        batch = c.shape[0]
        z = torch.randn((batch,self.latent_size)).to(c.device)
        recons_batch = self.decode(z, c).reshape(-1,self.feature_size,1)
        return recons_batch
    
    def forward(self, x, c):
        # x: (bs, feature_size, 1)
        # c: (bs, condition_size, 1)
        mu, logvar = self.encode(x, c, attention_type = self.attention_type)
        z = self.reparameterize(mu, logvar)
        recons_batch = self.decode(z, c).reshape(-1,self.feature_size,1)
        return recons_batch, mu, logvar