from typing import Tuple

import pytest
import torch

from bert.config import BertConfig
from bert.model import BertEmbedding, BertLayer, BertModel, BertMultiHeadAttention

BATCH_SIZE, SEQ_LENGTH = 32, 64
BertFeature = Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]


@pytest.fixture
def bert_config() -> BertConfig:
    config_dict = {
        "hidden_size": 768,
        "hidden_act": "gelu",
        "initializer_range": 0.02,
        "vocab_size": 30522,
        "hidden_dropout_prob": 0.1,
        "num_attention_heads": 12,
        "type_vocab_size": 2,
        "max_position_embeddings": 512,
        "num_hidden_layers": 12,
        "intermediate_size": 3072,
        "attention_probs_dropout_prob": 0.1,
        "layer_norm_eps": 1e-12,
    }
    return BertConfig(**config_dict)


@pytest.fixture
def bert_inputs(bert_config: BertConfig) -> BertFeature:
    input_ids = torch.randint(0, bert_config.vocab_size, (BATCH_SIZE, SEQ_LENGTH))
    attention_mask = input_ids.gt(0).float()
    token_type_ids = torch.randint(0, bert_config.type_vocab_size, (BATCH_SIZE, SEQ_LENGTH))
    position_ids = torch.arange(SEQ_LENGTH, dtype=torch.long)
    return input_ids, attention_mask, token_type_ids, position_ids


def test_bert_embedding(bert_config: BertConfig, bert_inputs: BertFeature):
    input_ids, _, token_type_ids, position_ids = bert_inputs

    embedding = BertEmbedding(bert_config)
    embed_output = embedding(input_ids, token_type_ids, position_ids)

    assert tuple(embed_output.size()) == (BATCH_SIZE, SEQ_LENGTH, bert_config.hidden_size)


def test_bert_multihead_attention(bert_config: BertConfig, bert_inputs: BertFeature):
    input_ids, attention_mask, token_type_ids, position_ids = bert_inputs

    embedding = BertEmbedding(bert_config)
    hidden_states = embedding(input_ids, token_type_ids, position_ids)

    multi_head_attention = BertMultiHeadAttention(bert_config)
    multi_head_attention_outputs = multi_head_attention.forward(hidden_states, attention_mask)

    assert tuple(multi_head_attention_outputs.size()) == (BATCH_SIZE, SEQ_LENGTH, bert_config.hidden_size)


def test_bert_layer(bert_config: BertConfig, bert_inputs: BertFeature):
    input_ids, attention_mask, token_type_ids, position_ids = bert_inputs

    embedding = BertEmbedding(bert_config)
    hidden_states = embedding(input_ids, token_type_ids, position_ids)

    bert_layer = BertLayer(bert_config)
    layer_output = bert_layer.forward(hidden_states, attention_mask)

    assert tuple(layer_output.size()) == (BATCH_SIZE, SEQ_LENGTH, bert_config.hidden_size)


def test_bert_model(bert_config: BertConfig, bert_inputs: BertFeature):
    bert = BertModel(bert_config)
    pooled_output, seq_output = bert(*bert_inputs)

    assert tuple(pooled_output.size()) == (BATCH_SIZE, bert_config.hidden_size)
    assert tuple(seq_output.size()) == (BATCH_SIZE, SEQ_LENGTH, bert_config.hidden_size)
