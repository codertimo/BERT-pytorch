from .dataset import WordVocab, BERTDatasetCreator

import argparse
import tqdm


def build():
    parser = argparse.ArgumentParser()
    parser.add_argument("-v", "--vocab_path", required=True, type=str)
    parser.add_argument("-c", "--corpus_path", required=True, type=str)
    parser.add_argument("-e", "--encoding", default="utf-8", type=str)
    parser.add_argument("-o", "--output_path", required=True, type=str)
    parser.add_argument("-w", "--workers", default=4, type=int)
    args = parser.parse_args()

    print("Loading Word Vocab", args.vocab_path)
    word_vocab = WordVocab.load_vocab(args.vocab_path)
    print("VOCAB SIZE=", len(word_vocab))

    builder = BERTDatasetCreator(corpus_path=args.corpus_path,
                                 vocab=word_vocab, seq_len=None,
                                 encoding=args.encoding)

    output_form = "%s\t%s\t%s\t%s\t%d\n"

    def work(index):
        data = builder[index]

        data["t1_random"], data["t2_random"] = [",".join([str(i) for i in t])
                                                for t in [data["t1_random"], data["t2_random"]]]

        data["t1_label"], data["t2_label"] = [",".join([str(i) for i in label])
                                              for label in [data["t1_label"], data["t2_label"]]]

        output = output_form % (data["t1_random"], data["t2_random"],
                                data["t1_label"], data["t2_label"], data["is_next"])
        return output

    with open(args.output_path, 'w', encoding=args.encoding) as f:
        for i in tqdm.tqdm(range(len(builder)), total=len(builder), desc="Building Dataset"):
            f.write(work(i))
