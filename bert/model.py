import math
from typing import Tuple

import torch
from torch import Tensor, nn
from torch.nn import functional as fnn

from .config import BertConfig


class BertEmbedding(nn.Module):
    def __init__(self, config: BertConfig):
        super().__init__()
        self.word_embeddings = nn.Embedding(config.vocab_size, config.hidden_size)
        self.token_type_embeddings = nn.Embedding(config.type_vocab_size, config.hidden_size)
        self.position_embeddings = nn.Embedding(config.max_position_embeddings, config.hidden_size)

        self.layer_norm = nn.LayerNorm(config.hidden_size, config.layer_norm_eps)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def forward(self, input_ids: Tensor, token_type_ids: Tensor, turn_type_ids: Tensor) -> Tensor:
        word_embeds = self.word_embeddings(input_ids)
        token_type_embeds = self.token_type_embeddings(token_type_ids)
        position_embed = self.position_embeddings(turn_type_ids)

        embed_output = word_embeds + token_type_embeds + position_embed
        embed_output = self.layer_norm(embed_output)
        embed_output = self.dropout(embed_output)
        return embed_output


class BertMultiHeadAttention(nn.Module):
    def __init__(self, config: BertConfig):
        super().__init__()
        self.num_attention_heads = config.num_attention_heads
        self.head_hidden_size = config.hidden_size // config.num_attention_heads
        self.hidden_size = config.hidden_size

        self.query = nn.Linear(config.hidden_size, config.hidden_size)
        self.key = nn.Linear(config.hidden_size, config.hidden_size)
        self.value = nn.Linear(config.hidden_size, config.hidden_size)
        self.attention_dropout = nn.Dropout(config.attention_probs_dropout_prob)

        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.layer_norm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def forward(self, hidden_states: Tensor, attention_mask: Tensor) -> Tensor:
        # query, key, value linear projection
        query_output = self.query(hidden_states)
        key_output = self.key(hidden_states)
        value_output = self.value(hidden_states)

        seq_len = hidden_states.size(1)

        # split hidden_state into num_heads pieces (hidden_size = num_attention_heads * head_hidden_size)
        # ops #1: (batch, seq_len, hidden_size) -> (batch, seq_len, num_attention_heads, head_hidden_size)
        # ops #2: (batch, seq_len, num_attention_heads, head_hidden_size) -> (batch, num_attention_heads, seq_len, head_hidden_size)
        # output: (batch, num_attention_heads, seq_len, head_hidden_size)
        query_output = query_output.view(-1, seq_len, self.num_attention_heads, self.head_hidden_size)
        query_output = query_output.transpose(1, 2)
        key_output = key_output.view(-1, seq_len, self.num_attention_heads, self.head_hidden_size)
        key_output = key_output.transpose(1, 2)
        value_output = value_output.view(-1, seq_len, self.num_attention_heads, self.head_hidden_size)
        value_output = value_output.transpose(1, 2)

        # attention_ops: (batch, num_attention_heads, seq_len, head_hidden_size) x (batch, num_attention_heads, head_hidden_size, seq_len)
        # output: (batch, num_attention_heads, seq_len, seq_len)
        attention_scores = torch.matmul(query_output, key_output.transpose(2, 3))
        attention_scores = attention_scores / math.sqrt(self.head_hidden_size)

        # TODO: attention mask
        # TODO: head mask

        # normalize attention scores to probs
        attention_probs = fnn.softmax(attention_scores, dim=-1)
        attention_probs = self.dropout(attention_probs)

        # context_ops: (batch, num_attention_heads, seq_len, seq_len) x (batch, num_attention_heads, seq_len, head_hidden_size)
        # output: (batch, num_attention_heads, seq_len, hidden_size)
        context_encoded_output = torch.matmul(attention_probs, value_output)

        # merge multi-head output to single head output
        # ops1: (batch, num_attention_heads, seq_len, head_hidden_size) -> (batch, seq_len, num_attention_heads, head_hidden_size)
        # ops2: (batch, seq_len, num_attention_heads, head_hidden_size) -> (batch, seq_len, hidden_size)
        # output: (batch, seq_len, num_attention_heads, head_hidden_size)
        context_encoded_output = context_encoded_output.transpose(1, 2).contiguous()
        context_encoded_output = context_encoded_output.view(-1, seq_len, self.hidden_size)

        # output linear projection + layer norm + dropout
        context_encoded_output = self.dense(context_encoded_output)
        context_encoded_output = self.layer_norm(context_encoded_output)
        context_encoded_output = self.dropout(context_encoded_output)

        return context_encoded_output


class BertLayer(nn.Module):
    def __init__(self, config: BertConfig):
        super().__init__()
        self.attention = BertMultiHeadAttention(config)

        self.intermediate_dense = nn.Linear(config.hidden_size, config.intermediate_size)
        self.intermediate_activation_fn = nn.GELU()

        self.output_dense = nn.Linear(config.intermediate_size, config.hidden_size)
        self.output_layer_norm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.output_dropout = nn.Dropout(config.hidden_dropout_prob)

    def forward(self, hidden_states: Tensor, attention_mask: Tensor) -> Tensor:
        context_encoded_output = self.attention(hidden_states, attention_mask)

        intermediate_output = self.intermediate_dense(context_encoded_output)
        intermediate_output = self.intermediate_activation_fn(intermediate_output)

        layer_output = self.output_dense(intermediate_output)
        layer_output = self.output_layer_norm(layer_output)
        layer_output = self.output_dropout(layer_output)
        return layer_output


class BertModel(nn.Module):
    def __init__(self, config: BertConfig):
        super().__init__()
        self.config = config

        self.embedding = BertEmbedding(config)
        self.layers = nn.ModuleList([BertLayer(config) for layer in range(config.num_hidden_layers)])

        self.pooler_dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.pooler_activation_fn = nn.Tanh()

    def forward(
        self, input_ids: Tensor, attention_mask: Tensor, token_type_ids: Tensor, position_ids: Tensor
    ) -> Tuple[Tensor, Tensor]:
        hidden_states = self.embedding(input_ids, token_type_ids, position_ids)

        for layer in self.layers:
            hidden_states = layer(hidden_states, attention_mask)

        pooled_output = self.pooler_dense(hidden_states[:, 0])
        pooled_output = self.pooler_activation_fn(pooled_output)

        return pooled_output, hidden_states
