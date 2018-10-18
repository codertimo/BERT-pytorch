from torch.utils.data import Dataset

import random
import tqdm


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
                    tokens[i] = self.vocab.mask_index

                # 10% randomly change token to random token
                elif 0.8 <= prob < 0.9:
                    tokens[i] = random.randrange(len(self.vocab))

                # 10% randomly change token to current token
                elif prob >= 0.9:
                    tokens[i] = self.vocab.stoi.get(token, self.vocab.unk_index)

                output_label.append(self.vocab.stoi.get(token, self.vocab.unk_index))

            else:
                tokens[i] = self.vocab.stoi.get(token, self.vocab.unk_index)
                output_label.append(0)

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
