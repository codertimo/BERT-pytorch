import torch.nn as nn
from torch.optim import Adam
from model import BERTLM, BERT


class BERTTrainer:
    def __init__(self, bert: BERT, vocab_size, train_dataloader, test_dataloader=None):
        self.bert = bert
        self.lm = BERTLM(bert, vocab_size)

        self.train_data = train_dataloader
        self.test_data = test_dataloader

        self.optim = Adam(self.lm.parameters())
        self.criterion = nn.NLLLoss()

    def train(self, epoch):
        self.iteration(epoch, self.train_data)

    def test(self, epoch):
        self.iteration(epoch, self.test_data, train=False)

    def iteration(self, epoch, data_loader, train=True):
        pass
