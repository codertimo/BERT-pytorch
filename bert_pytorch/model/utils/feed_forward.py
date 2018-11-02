import torch.nn as nn
from .gelu import GELU


class FeedForward(nn.Module):
    "Implements FFN equation."

    def __init__(self, d_model, d_ff):
        super(FeedForward, self).__init__()
        self.w_1 = nn.Linear(d_model, d_ff)
        self.w_2 = nn.Linear(d_ff, d_model)
        self.activation = GELU()

    def forward(self, x):
        x = self.w_1(x)
        x = self.activation(x)
        x = self.w_2(x)
        return x
