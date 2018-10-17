import torch
import torch.nn as nn
from torch.optim import Adam

from model import BERTLM, BERT

import tqdm
import os


class BERTTrainer:
    def __init__(self, bert: BERT, vocab_size, train_dataloader, test_dataloader=None,
                 lr=1e-4, betas=(0.9, 0.999), weight_decay=0.01):
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        self.bert = bert
        self.model = BERTLM(bert, vocab_size).to(self.device)

        if torch.cuda.device_count() > 1:
            print("Using %d GPUS for BERT" % torch.cuda.device_count())
            self.model = nn.DataParallel(self.model)

        self.train_data = train_dataloader
        self.test_data = test_dataloader

        self.optim = Adam(self.model.parameters(), lr=lr, betas=betas, weight_decay=weight_decay)
        self.criterion = nn.NLLLoss(ignore_index=0)
        print("Total Parameters:", sum([p.nelement() for p in self.model.parameters()]))

    def train(self, epoch):
        self.iteration(epoch, self.train_data)

    def test(self, epoch):
        self.iteration(epoch, self.test_data, train=False)

    def iteration(self, epoch, data_loader, train=True):
        str_code = "train" if train else "test"
        data_iter = tqdm.tqdm(enumerate(data_loader),
                              desc="EP_%s:%d" % (str_code, epoch),
                              total=len(data_loader),
                              bar_format="{l_bar}{r_bar}")

        avg_loss = 0.0
        total_correct = 0
        total_element = 0

        for i, data in data_iter:
            data = {key: value.to(self.device) for key, value in data.items()}

            next_sent_output, mask_lm_output = self.model.forward(data["bert_input"], data["segment_label"])
            next_loss = self.criterion(next_sent_output, data["is_next"])
            mask_loss = self.criterion(mask_lm_output.transpose(1, 2), data["bert_label"])
            loss = next_loss + mask_loss

            correct = next_sent_output.argmax(dim=-1).eq(data["is_next"]).sum().item()

            avg_loss += loss.item()
            total_correct += correct
            total_element += data["is_next"].nelement()

            if train:
                self.optim.zero_grad()
                loss.backward()
                self.optim.step()

            post_fix = {
                "epoch": epoch,
                "avg_loss": avg_loss / (i + 1),
                "avg_acc": total_correct / total_element * 100
            }
            data_iter.set_postfix(post_fix)

        print("EP%d_%s, avg_loss=" % (epoch, str_code), avg_loss / len(data_iter), "total_acc=",
              total_correct * 100.0 / total_element)

    def save(self, output_dir, epoch, file_name="bert_trained_ep%d.model"):
        if isinstance(self.model, nn.DataParallel):
            model = self.model.module
        else:
            model = self.model

        with open(os.path.join(output_dir, file_name % epoch), "wb") as f:
            torch.save(model.cpu(), f)

        model.to(self.device)
