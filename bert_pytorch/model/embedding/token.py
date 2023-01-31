import torch.nn as nn


class TokenEmbedding(nn.Embedding):
    """nn.Embedding class is the Parent class of TokenEmbedding
       nn.Embedding(vocab_size, embed_size) return vocab_size vector with dimension of embed_size;
       nn.Embedding's method forward(self, input: Tensor) -> Tensor以input中的元素为index返回对应的向量:
    """
    def __init__(self, vocab_size, embed_size=512):
        super().__init__(vocab_size, embed_size, padding_idx=0)
