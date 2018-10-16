from torch.utils.data import Dataset
import tqdm
import random
import torch


class BERTDataset(Dataset):
    def __init__(self, corpus_path, vocab, seq_len, encoding="utf-8", corpus_lines=None):
        self.vocab = vocab
        self.seq_len = seq_len

        self.datas = []
        with open(corpus_path, "r", encoding=encoding) as f:
            for line in tqdm.tqdm(f, desc="Loading Dataset", total=corpus_lines):
                t1, t2, t1_l, t2_l, is_next = line[:-1].split("\t")
                t1_l, t2_l = [[token for token in label.split(" ")] for label in [t1_l, t2_l]]
                is_next = int(is_next)
                self.datas.append({"t1": t1, "t2": t2, "t1_label": t1_l, "t2_label": t2_l, "is_next": is_next})

    def __len__(self):
        return len(self.datas)

    def __getitem__(self, item):
        # [CLS] tag = SOS tag, [SEP] tag = EOS tag
        t1 = self.vocab.to_seq(self.datas[item]["t1"], with_sos=True, with_eos=True)
        t2 = self.vocab.to_seq(self.datas[item]["t2"], with_eos=True)

        t1_label = [0] + self.vocab.to_seq(self.datas[item]["t1_label"]) + [0]
        t2_label = self.vocab.to_seq(self.datas[item]["t2_label"]) + [0]

        segment_label = ([1 for _ in range(len(t1))] + [2 for _ in range(len(t2))])[:self.seq_len]
        bert_input = (t1 + t2)[:self.seq_len]
        bert_label = (t1_label + t2_label)[:self.seq_len]

        padding = [self.vocab.pad_index for _ in range(self.seq_len - len(t1) - len(t2))]
        bert_input.extend(padding), bert_label.extend(padding), segment_label.extend(padding)

        output = {"bert_input": bert_input,
                  "bert_label": bert_label,
                  "segment_label": segment_label,
                  "is_next": self.datas[item]["is_next"]}

        return {key: torch.tensor(value) for key, value in output.items()}


class BERTDatasetCreator(Dataset):
    def __init__(self, corpus_path, vocab, seq_len, encoding="utf-8"):
        self.vocab = vocab
        self.seq_len = seq_len

        with open(corpus_path, "r", encoding=encoding) as f:
            self.datas = [line[:-1].split("\t") for line in tqdm.tqdm(f, desc="Loading Dataset")]

    def random_word(self, sentence):
        tokens = sentence.split()
        output_label = []

        for i, token in enumerate(tokens):
            prob = random.random()
            if prob < 0.15:
                prob = random.random()

                # 80% randomly change token to make token
                if prob < 0.8:
                    tokens[i] = "<mask>"

                # 10% randomly change token to random token
                elif 0.8 <= prob < 0.9:
                    tokens[i] = self.vocab.itos[random.randrange(len(self.vocab))]

                # 10% randomly change token to current token
                elif prob >= 0.9:
                    pass

                output_label.append(token)

            else:
                output_label.append("<pad>")

        return tokens, output_label

    def random_sent(self, index):
        # output_text, label(isNotNext:0, isNext:1)
        if random.random() > 0.5:
            return self.datas[index][1], 1
        else:
            return self.datas[random.randrange(len(self.datas))][1], 0

    def __getitem__(self, index):
        t1, (t2, is_next_label) = self.datas[index][0], self.random_sent(index)
        t1_random, t1_label = self.random_word(t1)
        t2_random, t2_label = self.random_word(t2)

        return {"t1_random": t1_random, "t2_random": t2_random,
                "t1_label": t1_label, "t2_label": t2_label,
                "is_next": is_next_label}

    def __len__(self):
        return len(self.datas)
