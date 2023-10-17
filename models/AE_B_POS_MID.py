import torch
from torch import nn
import math

class MLP(nn.Module):
    def __init__(self,input_dim,hidden_dim,output_dim) -> None:
        super(MLP,self).__init__()

        self.linear1 = nn.Linear(input_dim, hidden_dim)
        self.relu = nn.LeakyReLU()
        self.linear2 = nn.Linear(hidden_dim, hidden_dim)  # 2个隐层
        self.relu2 = nn.LeakyReLU()
        self.linear3 = nn.Linear(hidden_dim, output_dim)
    
    def forward(self,x):
        
        x = self.linear1(x)
        x = self.relu(x)
        x = self.linear2(x)
        x = self.relu2(x)
        x = self.linear3(x)
        return x


class AE(nn.Module):
    def __init__(self, in_channels=200*16):
        super(AE, self).__init__()
        self.encoder=nn.Sequential(
            nn.Linear(in_channels,256),
            nn.LeakyReLU(),
            nn.Linear(256,64),
            nn.LeakyReLU(),
            nn.Linear(64,20),
            nn.LeakyReLU()
        )
        self.decoder=nn.Sequential(
            nn.Linear(20,64),
            nn.LeakyReLU(),
            nn.Linear(64,256),
            nn.LeakyReLU(),
            nn.Linear(256,in_channels),
            nn.Sigmoid()
        )
    def forward(self,x):       
        encoded=self.encoder(x)
        decoded=self.decoder(encoded)
        return decoded

def add_position_embedding(x):
    bs, seq_len, _ = x.shape
    d_model = x.shape[-1]

    # 生成位置编码
    position_encoding = torch.zeros(seq_len, d_model)
    position = torch.arange(0, seq_len, dtype=torch.float).unsqueeze(1)
    div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
    position_encoding[:, 0::2] = torch.sin(position * div_term)
    position_encoding[:, 1::2] = torch.cos(position * div_term)
    position_encoding = position_encoding.to(x.device)
    # 将位置编码与输入x相加
    x_with_position = x + position_encoding.unsqueeze(0)

    return x_with_position

class AE_B_POS_MID(nn.Module):
    def __init__(self,in_channels=2,
                 ae_channels=39*16,
                ) -> None:
        super().__init__()

        self.mlp11 = MLP(in_channels,8,16) # (B,20,2) --> (B,20,16)
        self.mlp12 = MLP(ae_channels,ae_channels//2,ae_channels)
        self.ae = AE(in_channels=ae_channels)
        self.mlp21 = MLP(ae_channels, ae_channels*2, 200*16)
        self.mlp22 = MLP(16, 8, in_channels)

    def calMidLine(self, data):
        n = data.shape[1] // 2
        return torch.stack([data[:,1:n, 0], (data[:,1:n, 1] + data[:,-n+1:, 1].flip(1)) / 2], dim=2)
    
    def forward(self,x,p,mid_input): # p就是物理参数,mid_input是中弧线的点
        bs = x.shape[0]
        # x = add_position_embedding(x)
        # mid_input = add_position_embedding(mid_input)
        x = torch.cat((x,mid_input,p),dim=1) # (B,39,2)
        ae_input = self.mlp11(x) # (B,30,2) --> (B,30,16)
        ae_input = ae_input.reshape(bs,-1) # (B,30,16) --> (B,30*16)
        ae_input = self.mlp12(ae_input) # (B,30*16) --> (B,30*16)
        ae_output = self.ae(ae_input) # (B,30*16) --> (B,30*16)
        ae_output = self.mlp21(ae_output) # (B,30*16) --> (B,200*16)
        y = ae_output.reshape(bs,200,16) # (B,200*16) --> (B,200,16)

        output = self.mlp22(y) # (B,200,16) --> (B,200,2)
        return output,self.calMidLine(output) # (B,200,2),(B,99,2)