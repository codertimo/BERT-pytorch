from dataset.dataset import BERTDatasetCreator
from dataset import WordVocab
from torch.utils.data import DataLoader
import argparse
import tqdm

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
        output_form = "%s\t%s\t%s\t%s\t%d\n"
        t1_text, t2_text = [" ".join(t) for t in [data["t1_random"], data["t2_random"]]]
        t1_label, t2_label = [" ".join([str(i) for i in label]) for label in [data["t1_label"], data["t2_label"]]]
        output = output_form % (t1_text, t2_text, t1_label, t2_label, data["is_next"])
        f.write(output)
