from torch.utils.data import Dataset
import tqdm
import random
import argparse
import json


class BERTDataset(Dataset):
    def __init__(self, corpus_path, vocab, seq_len, encoding="utf-8"):
        self.vocab = vocab
        self.seq_len = seq_len

        with open(corpus_path, "r", encoding=encoding) as f:
            self.datas = [line[:-1].split("\t") for line in tqdm.tqdm(f, desc="Loading Dataset")]

    def __len__(self):
        return len(self.datas)

    def __getitem__(self, item):
        # [CLS] tag = SOS tag, [SEP] tag = EOS tag
        t1, t1_len = self.vocab.to_seq(self.datas[item][0], seq_len=self.seq_len, with_sos=True, with_eos=True)
        t2, t2_len = self.vocab.to_seq(self.datas[item][1], seq_len=self.seq_len, with_eos=True)

        return {"t1": t1, "t2": t2, "t1_len": t1_len, "t2_len": t2_len}


class BERTDatasetCreator(BERTDataset):
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
        if random.random() > 0.5:
            return self.datas[index][2]
        else:
            return self.datas[random.randrange(len(self.datas))][2]

    def __getitem__(self, index):
        t1, t2 = self.datas[index], self.random_sent(index)
        t1_random, t1_label = self.random_word(t1)
        t2_random, t2_label = self.random_word(t2)

        return {"t1_random": t1_random, "t2_random": t2_random, "t1_label": t1_label, "t2_label": t2_label}


if __name__ == "__main__":
    from .vocab import WordVocab

    parser = argparse.ArgumentParser()
    parser.add_argument("-v", "--vocab_path", required=True, type=str)
    parser.add_argument("-c", "--corpus_path", required=True, type=str)
    parser.add_argument("-e", "--encoding", default="utf-8", type=str)
    parser.add_argument("-o", "--output_path", required=True, type=str)
    args = parser.parse_args()

    vocab = WordVocab.load_vocab(args.vocab_path)
    builder = BERTDatasetCreator(corpus_path=args.corpus_path, vocab=vocab, seq_len=None, encoding=args.encoding)

    with open(args.output_path, 'w', encoding=args.encoding) as f:
        for index in range(len(builder)):
            data = builder[index]
            # todo : end of coding today


