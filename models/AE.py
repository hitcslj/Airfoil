import torch
from torch import nn

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
        
class AEBackboneNet(nn.Module):
    def __init__(self,in_channels=2,
                 ae_channels=200*16,
                ) -> None:
        super().__init__()

        self.mlp11 = MLP(in_channels,8,16) # (B,200,2) --> (B,200,16)
        self.mlp12 = MLP(ae_channels,ae_channels//2,ae_channels)
        self.ae = AE(in_channels=ae_channels)
        self.mlp21 = MLP(16, 8, in_channels)

    def forward(self,x):
        bs = x.shape[0]
        ae_input = self.mlp11(x) # (B,200,2) --> (B,200,16)
        ae_input = ae_input.reshape(bs,-1) # (B,200,16) --> (B,200*16)
        ae_input = self.mlp12(ae_input) # (B,200*16) --> (B,200*16)
        ae_output = self.ae(ae_input) # (B,200*16) --> (B,200*16)
        y = ae_output.reshape(bs,200,16) # (B,200*16) --> (B,200,16)
        output = self.mlp21(y) # (B,200,16) --> (B,200,2)
        return {'ae_output':ae_output,'ae_input':ae_input,'output':output}