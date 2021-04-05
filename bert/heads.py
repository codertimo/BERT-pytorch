from typing import Tuple

import torch
from torch import nn

from .config import BertConfig
from .model import BertModel


class BertPretrainingHeads(nn.Module):
    def __init__(self, config: BertConfig, bert: BertModel):
        super().__init__()
        self.bert = bert
        self.language_model_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        self.language_model_head.bias = nn.Parameter(torch.zeros(config.vocab_size))

        self.next_sentence_prediction_head = nn.Linear(config.hidden_size, 2)

    def forward(self, *bert_input_args, **bert_input_kwargs) -> Tuple[torch.Tensor, torch.Tensor]:
        pooled_output, seq_output = self.bert(*bert_input_args, **bert_input_kwargs)
        lm_output = self.language_model_head(seq_output)
        nsp_output = self.next_sentence_prediction_head(pooled_output)
        return lm_output, nsp_output
