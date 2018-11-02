import torch.nn as nn


class PositionalEmbedding(nn.Embedding):

    def __init__(self, d_model, max_len=512):
        super().__init__(max_len, d_model)

    def forward(self, x):
        return self.weight.data[:x.size(1)]
