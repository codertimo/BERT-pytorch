import argparse

from torch.utils.data import DataLoader

from model import BERT
from trainer import BERTTrainer
from dataset import BERTDataset, WordVocab

parser = argparse.ArgumentParser()

parser.add_argument("-d", "--train_dataset", required=True, type=str)
parser.add_argument("-t", "--test_dataset", type=str, default=None)
parser.add_argument("-v", "--vocab_path", required=True, type=str)
parser.add_argument("-o", "--output_dir", required=True, type=str)

parser.add_argument("-hs", "--hidden", type=int, default=128)
parser.add_argument("-n", "--layers", type=int, default=2)
parser.add_argument("-a", "--attn_heads", type=int, default=4)
parser.add_argument("-s", "--seq_len", type=int, default=10)

parser.add_argument("-b", "--batch_size", type=int, default=64)
parser.add_argument("-e", "--epochs", type=int, default=10)

args = parser.parse_args()

vocab = WordVocab.load_vocab(args.vocab_path)

print("Loading Train Dataset", args.train_dataset)
train_dataset = BERTDataset(args.train_dataset, vocab, seq_len=args.seq_len)
print("Loading Test Dataset", args.test_dataset)
test_dataset = BERTDataset(args.test_dataset, vocab, seq_len=args.seq_len) if args.test_dataset is not None else None

train_data_loader = DataLoader(train_dataset, batch_size=16)
test_data_loader = DataLoader(test_dataset) if test_dataset is not None else None

bert = BERT(len(vocab), hidden=args.hidden, n_layers=args.layers, attn_heads=args.attn_heads)
trainer = BERTTrainer(bert, len(vocab), train_dataloader=train_data_loader, test_dataloader=test_data_loader)

for epoch in range(args.epochs):
    trainer.train(epoch)
    trainer.save(args.output_dir, epoch)

    if test_data_loader is not None:
        trainer.test(epoch)
