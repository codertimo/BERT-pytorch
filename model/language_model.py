from .bert import BERT
import torch.nn as nn


class BERTLM(nn.Module):
    def __init__(self, bert: BERT, vocab_size):
        super().__init__()
        self.next_sentence = BERTNextSentence(bert)
        self.mask_lm = BERTMaskLM(bert, vocab_size)

    def forward(self, x):
        return self.next_sentence(x), self.mask_lm(x)


class BERTNextSentence(nn.Module):
    def __init__(self, bert: BERT):
        super().__init__()
        self.bert = bert
        self.linear = nn.Linear(self.bert.hidden, 2)
        self.softmax = nn.LogSoftmax(dim=-1)

    def forward(self, x):
        return self.softmax(self.linear(x))


class BERTMaskLM(nn.Module):
    def __init__(self, bert: BERT, vocab_size):
        super().__init__()
        self.bert = bert
        self.linear = nn.Linear(self.bert.hidden, vocab_size)
        self.softmax = nn.LogSoftmax(dim=-1)

    def forward(self, x):
        return self.softmax(self.linear(x))
