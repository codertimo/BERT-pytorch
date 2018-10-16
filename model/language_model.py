from .bert import BERT
import torch.nn as nn


class BERTLM(nn.Module):
    def __init__(self, bert: BERT, vocab_size):
        super().__init__()
        self.bert = bert
        self.next_sentence = BERTNextSentence(self.bert.hidden)
        self.mask_lm = BERTMaskLM(self.bert.hidden, vocab_size)

    def forward(self, x, segment_label):
        x = self.bert(x, segment_label)
        return self.next_sentence(x), self.mask_lm(x)


class BERTNextSentence(nn.Module):
    def __init__(self, hidden):
        super().__init__()
        self.linear = nn.Linear(hidden, 2)
        self.softmax = nn.LogSoftmax(dim=-1)

    def forward(self, x):
        return self.softmax(self.linear(x[:, 0]))


class BERTMaskLM(nn.Module):
    def __init__(self, hidden, vocab_size):
        super().__init__()
        self.linear = nn.Linear(hidden, vocab_size)
        self.softmax = nn.LogSoftmax(dim=-1)

    def forward(self, x):
        return self.softmax(self.linear(x))
