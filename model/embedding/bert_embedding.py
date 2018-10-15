import torch.nn as nn
from .token import TokenEmbedding
from .position import PositionalEmbedding
from .segment import SegmentEmbedding


class BERTEmbedding(nn.Module):
    def __init__(self, token_embedding: TokenEmbedding, dropout=0.1):
        super().__init__()
        self.token = token_embedding
        self.position = PositionalEmbedding(token_embedding.embedding_dim, dropout=dropout)
        self.segment = SegmentEmbedding(embed_size=token_embedding.embedding_dim)

    def forward(self, sequence):
        return self.token(sequence) + self.position(sequence), self.segment(sequence)
