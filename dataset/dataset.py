from torch.utils.data import Dataset
import tqdm
import torch


class BERTDataset(Dataset):
    def __init__(self, corpus_path, vocab, seq_len, encoding="utf-8", corpus_lines=None):
        self.vocab = vocab
        self.seq_len = seq_len

        self.datas = []
        with open(corpus_path, "r", encoding=encoding) as f:
            for line in tqdm.tqdm(f, desc="Loading Dataset", total=corpus_lines):
                t1, t2, t1_l, t2_l, is_next = line[:-1].split("\t")
                t1, t2 = [[int(token) for token in t.split(",")] for t in [t1, t2]]
                t1_l, t2_l = [[int(token) for token in label.split(",")] for label in [t1_l, t2_l]]
                is_next = int(is_next)
                self.datas.append({"t1": t1, "t2": t2, "t1_label": t1_l, "t2_label": t2_l, "is_next": is_next})

    def __len__(self):
        return len(self.datas)

    def __getitem__(self, item):
        # [CLS] tag = SOS tag, [SEP] tag = EOS tag
        t1 = [self.vocab.sos_index] + self.datas[item]["t1"] + [self.vocab.eos_index]
        t2 = self.datas[item]["t2"] + [self.vocab.eos_index]

        t1_label = [0] + self.datas[item]["t1_label"] + [0]
        t2_label = self.datas[item]["t2_label"] + [0]

        segment_label = ([1 for _ in range(len(t1))] + [2 for _ in range(len(t2))])[:self.seq_len]
        bert_input = (t1 + t2)[:self.seq_len]
        bert_label = (t1_label + t2_label)[:self.seq_len]

        padding = [self.vocab.pad_index for _ in range(self.seq_len - len(bert_input))]
        bert_input.extend(padding), bert_label.extend(padding), segment_label.extend(padding)

        output = {"bert_input": bert_input,
                  "bert_label": bert_label,
                  "segment_label": segment_label,
                  "is_next": self.datas[item]["is_next"]}

        return {key: torch.tensor(value) for key, value in output.items()}
