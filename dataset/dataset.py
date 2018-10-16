from torch.utils.data import Dataset
import tqdm
import random
import argparse
import torch


class BERTDataset(Dataset):
    def __init__(self, corpus_path, vocab, seq_len, encoding="utf-8"):
        self.vocab = vocab
        self.seq_len = seq_len

        self.datas = []
        with open(corpus_path, "r", encoding=encoding) as f:
            for line in tqdm.tqdm(f, desc="Loading Dataset"):
                t1, t2, t1_l, t2_l, is_next = line[:-1].split("\t")
                t1_l, t2_l = [[int(i) for i in label.split(",")] for label in [t1_l, t2_l]]
                is_next = int(is_next)
                self.datas.append({
                    "t1": t1,
                    "t2": t2,
                    "t1_label": t1_l,
                    "t2_label": t2_l,
                    "is_next": is_next
                })

    def __len__(self):
        return len(self.datas)

    def __getitem__(self, item):
        # [CLS] tag = SOS tag, [SEP] tag = EOS tag
        t1, t1_len = self.vocab.to_seq(self.datas[item]["t1"], seq_len=self.seq_len, with_sos=True, with_eos=True)
        t2, t2_len = self.vocab.to_seq(self.datas[item]["t2"], seq_len=self.seq_len, with_eos=True)

        output = {"t1": t1, "t2": t2,
                  "t1_len": t1_len, "t2_len": t2_len,
                  "t1_label": self.datas[item]["t1_label"], "t2_label": self.datas[item]["t2_label"],
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
            return self.datas[index][2], 1
        else:
            return self.datas[random.randrange(len(self.datas))][2], 0

    def __getitem__(self, index):
        t1, (t2, is_next_label) = self.datas[index], self.random_sent(index)
        t1_random, t1_label = self.random_word(t1)
        t2_random, t2_label = self.random_word(t2)

        return {"t1_random": t1_random, "t2_random": t2_random,
                "t1_label": t1_label, "t2_label": t2_label,
                "is_next": is_next_label}


if __name__ == "__main__":
    from .vocab import WordVocab

    parser = argparse.ArgumentParser()
    parser.add_argument("-v", "--vocab_path", required=True, type=str)
    parser.add_argument("-c", "--corpus_path", required=True, type=str)
    parser.add_argument("-e", "--encoding", default="utf-8", type=str)
    parser.add_argument("-o", "--output_path", required=True, type=str)
    args = parser.parse_args()

    word_vocab = WordVocab.load_vocab(args.vocab_path)
    builder = BERTDatasetCreator(corpus_path=args.corpus_path, vocab=word_vocab, seq_len=None, encoding=args.encoding)

    with open(args.output_path, 'w', encoding=args.encoding) as f:
        for index in tqdm.tqdm(range(len(builder)), desc="Building Dataset", total=len(builder)):
            data = builder[index]
            output_form = "%s\t%s\t%s\t%d\n"
            t1_text, t2_text = [" ".join(t) for t in [data["t1_random"], data["t2_random"]]]
            t1_label, t2_label = [",".join([str(i) for i in label]) for label in [data["t1_label"], data["t2_label"]]]
            output = output_form % (t1_text, t2_text, t1_label, t2_label, data["is_next"])
            f.write(output_form)
