import argparse

from torch.utils.data import DataLoader

from .model import BERT
from .trainer import BERTTrainer
from .dataset import BERTDataset, WordVocab


def train():
    parser = argparse.ArgumentParser()

    parser.add_argument("-c", "--train_dataset", required=True, type=str)
    parser.add_argument("-t", "--test_dataset", type=str, default=None)
    parser.add_argument("-v", "--vocab_path", required=True, type=str)
    parser.add_argument("-o", "--output_path", required=True, type=str)

    parser.add_argument("-hs", "--hidden", type=int, default=256)
    parser.add_argument("-l", "--layers", type=int, default=8)
    parser.add_argument("-a", "--attn_heads", type=int, default=8)
    parser.add_argument("-s", "--seq_len", type=int, default=20)

    parser.add_argument("-b", "--batch_size", type=int, default=64)
    parser.add_argument("-e", "--epochs", type=int, default=10)
    parser.add_argument("-w", "--num_workers", type=int, default=5)
    parser.add_argument("--with_cuda", type=bool, default=True)
    parser.add_argument("--log_freq", type=int, default=10)
    parser.add_argument("--corpus_lines", type=int, default=None)

    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--adam_weight_decay", type=float, default=0.01)
    parser.add_argument("--adam_beta1", type=float, default=0.9)
    parser.add_argument("--adam_beta2", type=float, default=0.999)

    args = parser.parse_args()

    print("Loading Vocab", args.vocab_path)
    vocab = WordVocab.load_vocab(args.vocab_path)
    print("Vocab Size: ", len(vocab))

    print("Loading Train Dataset", args.train_dataset)
    train_dataset = BERTDataset(args.train_dataset, vocab, seq_len=args.seq_len, corpus_lines=args.corpus_lines)

    print("Loading Test Dataset", args.test_dataset)
    test_dataset = BERTDataset(args.test_dataset, vocab,
                               seq_len=args.seq_len) if args.test_dataset is not None else None

    print("Creating Dataloader")
    train_data_loader = DataLoader(train_dataset, batch_size=args.batch_size, num_workers=args.num_workers)
    test_data_loader = DataLoader(test_dataset, batch_size=args.batch_size, num_workers=args.num_workers) \
        if test_dataset is not None else None

    print("Building BERT model")
    bert = BERT(len(vocab), hidden=args.hidden, n_layers=args.layers, attn_heads=args.attn_heads)

    print("Creating BERT Trainer")
    trainer = BERTTrainer(bert, len(vocab), train_dataloader=train_data_loader, test_dataloader=test_data_loader,
                          lr=args.lr, betas=(args.adam_beta1, args.adam_beta2), weight_decay=args.adam_weight_decay,
                          with_cuda=args.with_cuda, log_freq=args.log_freq)

    print("Training Start")
    for epoch in range(args.epochs):
        trainer.train(epoch)
        trainer.save(epoch, args.output_path)

        if test_data_loader is not None:
            trainer.test(epoch)
