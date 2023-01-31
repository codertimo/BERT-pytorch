import torch.nn as nn
import torch
import math


class PositionalEmbedding(nn.Module):

    def __init__(self, d_model, max_len=512):
        super().__init__()

        # Compute the positional encodings once in log space.
        pe = torch.zeros(max_len, d_model).float()
        pe.require_grad = False

        position = torch.arange(0, max_len).float().unsqueeze(1) #论文编码公式的分子
        div_term = (torch.arange(0, d_model, 2).float() * -(math.log(10000.0) / d_model)).exp() #论文编码公式的分母，先取log，再exp；渐少计算量？

        pe[:, 0::2] = torch.sin(position * div_term)

        #pe[:, 1::2].size(-1) is less than div_term.size(-1) when d_model is an odd number
        if pe[:, 1::2].size(-1) >= div_term.size(-1):
            pe[:, 1::2] = torch.cos(position * div_term)
        else:
            cos_len = pe[:, 1::2].size(-1)
            pe[:, 1::2] = torch.cos(position * div_term[:cos_len])

        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        return self.pe[:, :x.size(1)]
