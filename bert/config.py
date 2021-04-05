from typing import NamedTuple


class BertConfig(NamedTuple):
    vocab_size: int
    type_vocab_size: int
    max_position_embeddings: int

    hidden_size: int
    hidden_act: str
    initializer_range: float
    intermediate_size: int
    num_attention_heads: int
    num_hidden_layers: int

    layer_norm_eps: float
    hidden_dropout_prob: float
    attention_probs_dropout_prob: float
