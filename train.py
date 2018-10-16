import argparse
from dataset.dataset import BERTDataset, WordVocab
from torch.utils.data import DataLoader
from model import BERT, BERTLM

parser = argparse.ArgumentParser()
parser.add_argument("-d", "--dataset_path", required=True, type=str)
parser.add_argument("-v", "--vocab_path", required=True, type=str)
args = parser.parse_args()

vocab = WordVocab.load_vocab(args.vocab_path)
dataset = BERTDataset(args.dataset_path, vocab, seq_len=10)
data_loader = DataLoader(dataset, batch_size=16)

bert = BERT(len(vocab), hidden=128, n_layers=2, attn_heads=4)


for data in data_loader:
    x = model.forward(data["t1"])
    print(x.size())
