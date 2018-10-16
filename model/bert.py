import torch.nn as nn

from model.transformer import TransformerBlock
from model.embedding import BERTEmbedding


class BERT(nn.Module):
    def __init__(self, vocab_size, hidden=768, n_layers=12, attn_heads=12, dropout=0.1):
        super().__init__()
        self.hidden = hidden
        self.n_layers = n_layers
        self.attn_heads = attn_heads
        self.feed_forward_hidden = hidden * 4

        self.embedding = BERTEmbedding(vocab_size=vocab_size, embed_size=hidden)
        self.transformer_blocks = nn.ModuleList(
            [TransformerBlock(hidden=hidden,
                              attn_heads=attn_heads,
                              feed_forward_hidden=hidden * 4,
                              dropout=dropout)
             for _ in range(n_layers)])

    def forward(self, x):
        mask = (x > 0).unsqueeze(1).repeat(1, x.size(1), 1)

        # sequence -> embedding : (batch_size, seq_len) -> (batch_size, seq_len, embed_size)
        x = self.embedding(x)

        # embedding through the transformer self-attention
        # embedding (batch_size, seq_len, embed_size = hidden) -> transformer_output (batch_size, seq_len, hidden)
        # loop transformer (batch_size, seq_len, hidden) -> transformer_output (batch_size, seq_len, hidden)
        for transformer in self.transformer_blocks:
            x = transformer.forward(x, mask)

        return x
