import json
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

    hidden_dropout_prob: float
    attention_probs_dropout_prob: float
    layer_norm_eps: float = 1e-12

    @classmethod
    def from_json(cls, config_path: str) -> "BertConfig":
        with open(config_path) as f:
            config_dict = json.load(f)
            return BertConfig(**config_dict)
