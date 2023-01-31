import torch.nn as nn
import torch.nn.functional as F
import torch

import math


class Attention(nn.Module):
    """
    Compute 'Scaled Dot Product Attention
    """

    def forward(self, query, key, value, mask=None, dropout=None):
        scores = torch.matmul(query, key.transpose(-2, -1)) \
                 / math.sqrt(query.size(-1))

        if mask is not None:
            #transformer中的mask的作用：encoder中是去除<pad>序列的影响；decoder中是去除'不可见逻辑'
            #这里显然是前者；
            scores = scores.masked_fill(mask == 0, -1e9) #注意mask和score需要是可广播的

        p_attn = F.softmax(scores, dim=-1)

        if dropout is not None:
            p_attn = dropout(p_attn)

        return torch.matmul(p_attn, value), p_attn
