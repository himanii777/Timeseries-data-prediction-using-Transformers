import torch
import torch.nn as nn
import math


# we dont need "embedding" since our data are already numbers.

class PositionalEncoding(nn.Module):

    def __init__(self, d_model:int,seq_len=5000) -> None:
        super().__init__()
        self.dmodel= d_model
        self.seq_len= seq_len
        # self.batch_size= batch_size
        self.pe= torch.zeros(seq_len, d_model)
        position = torch.arange(0, seq_len, dtype=torch.float).unsqueeze(1)
        div_term= torch.exp(torch.arange(0,d_model,2).float()*(-math.log(10000)/d_model))

        self.pe[:,0::2]= torch.sin(position*div_term)
        self.pe[:,1::2]= torch.cos(position*div_term)

        self.pe = self.pe.unsqueeze(0) #(1, dmodel, seq_len)

        self.register_buffer('p', self.pe)

    def forward(self, x):
          x = x + self.p[:, :x.size(1)].to(x.device)
          return x


class TransformerModel(nn.Module):
    def __init__(self, seq_len=256, d_model=512, nhead=4, num_layers=3):
        super(TransformerModel, self).__init__()
        self.encoder = nn.Linear(seq_len, d_model)
        self.pos_encoder = PositionalEncoding(d_model=d_model,)
        encoder_layers = nn.TransformerEncoderLayer(d_model, nhead)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layers, num_layers)
        self.decoder = nn.Linear(d_model, seq_len)

    def forward(self, x):
        x= self.encoder(x)
        x = self.pos_encoder(x)
        x = self.transformer_encoder(x)
        x = self.decoder(x)
        return x

