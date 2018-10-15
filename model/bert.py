import torch.nn as nn

from model.transformer import TransformerBlock
from model.embedding import BERTEmbedding


class BERT(nn.Module):
    def __init__(self, embedding: BERTEmbedding, hidden=768, n_layers=12, attn_heads=12, dropout=0.1):
        super().__init__()
        self.hidden = hidden
        self.n_layers = n_layers
        self.attn_heads = attn_heads
        self.feed_forward_hidden = hidden * 4

        self.embedding: BERTEmbedding = embedding
        self.transformer_blocks = nn.ModuleList([TransformerBlock(hidden=hidden,
                                                                  attn_heads=attn_heads,
                                                                  feed_forward_hidden=self.feed_forward_hidden,
                                                                  dropout=dropout)
                                                 for _ in range(n_layers)])

    def forward(self, x, mask=None):
        for transformer in self.transformer_blocks:
            x = transformer.forward(x, mask)
        return x
