import os
import pickle
import random
from argparse import ArgumentParser, Namespace
from functools import partial
from glob import glob
from multiprocessing import Pool, cpu_count

from tokenizers import BertWordPieceTokenizer, Tokenizer
from tokenizers.trainers import WordPieceTrainer
from tqdm import tqdm

from bert.pretrain.feature import load_corpus, make_bert_pretrain_feature, tokenize_document
from bert.utils import get_logger

# fmt: off
parser = ArgumentParser()
parser.add_argument("--corpus-paths", type=str, required=True, help="corpus paths (glob style)")
parser.add_argument("--output-dir", type=str, required=True, help="output directory")
parser.add_argument("--max-seq-length", default=128, type=int, help="maximum sequence length of input features")
parser.add_argument("--short-seq-prob", default=0.2, type=float, help="probability to make shorter sequence")
parser.add_argument("--masked-lm-prob", default=0.15, type=float, help="masking prob")
parser.add_argument("--max-prediction-per-seq", default=20, type=int, help="maximum masked tokens per input")
parser.add_argument("--num-duplicates", default=10, type=int, help="number of dumplication for each document")
parser.add_argument("--num-workers", default=-1, type=int, help="num worker to multi-process (default: number of CㅈPU cores)")
parser.add_argument("--num-features-per-file", default=100000, type=int, help="number of features to save on single file")
parser.add_argument("--vocab-size", default=50004, type=int, help="vocab size of tokenizer")
parser.add_argument("--pretrained-tokenizer-json-path", type=str, help="(optional) pretrained tokenizer json path, don't give anything to pretrained-tokenizer-vocab-{path, is-lowercase}")
parser.add_argument("--pretrained-tokenizer-vocab-path", type=str, help="(optional) pretrained tokenizer vocab path")
parser.add_argument("--pretrained-tokenizer-is-uncased", action="store_true", help="(optional, default:False) if pretrained-tokenizer-vocab is not None")
# fmt: on

logger = get_logger("BERT-data")


def main(args: Namespace) -> int:
    documents = []
    for corpus_path in glob(args.corpus_paths):
        logger.info(f"[+] Parsing corpus: {corpus_path}")
        documents.extend(load_corpus(corpus_path))

    if args.pretrained_tokenizer_json_path:
        logger.info(f"[+] Loading WordPieceTokenizer from {args.pretrained_tokenizer_json_path}")
        tokenizer = Tokenizer.from_file(args.pretrained_tokenizer_json_path)
    elif args.pretrained_tokenizer_vocab_path:
        logger.info(f"[+] Loading WordPieceTokenizer from {args.pretraeind_vocab_path}")
        tokenizer = BertWordPieceTokenizer(
            args.pretrained_tokenizer_vocab_path, lowercase=not args.pretrained_tokenizer_is_cased
        )
    else:
        logger.info(f"[+] Training WordPieceTokenizer with {args.corpus_paths}")
        tokenizer = BertWordPieceTokenizer()
        trainer = WordPieceTrainer(vocab_size=args.vocab_size, min_frequency=1)
        trainer.train(glob(args.corpus_paths))

        trained_tokenizer_path = os.path.join(args.output_dir, "tokenizer.json")
        logger.info(f"[+] Saving trained WordPieceTokeizer to {trained_tokenizer_path}")
        tokenizer.save(trained_tokenizer_path, pretty=True)

    logger.info(f"[+] Tokenizing {len(documents)} documents start")
    num_worker = args.num_worker if args.num_worker else cpu_count()
    with Pool(num_worker) as pool:
        tokenize_fn = partial(tokenize_document, tokenizer=tokenizer)
        list(tqdm(pool.imap_unordered(tokenize_fn, documents), total=len(documents), desc="tokenizing"))

    logger.info("[+] Generating random documents for negative NSP")

    def random_document_generator():
        for document_index in range(len((documents))):
            random_index = random.randint(0, len(documents))
            while random_index == document_index:
                random_index = random.randint(0, len(documents))
            yield documents[random_index]

    random_documents = [random_document for random_document in random_document_generator()]

    logger.info("[+] Making BERT pre-training input features")
    for i in range(0, len(documents), args.num_features_per_file):
        output_path = os.path.join(args.output_dir, "bert_pretraining_features.{i:08}.records")
        num_features = min(args.num_features_per_file, len(documents) - i)
        logger.info(f"[+] Writing {num_features} features {output_path}")

        documents_chunk = documents[i : i + num_features]
        random_documents_chunk = random_documents[i : i + num_features]

        with Pool(num_worker) as pool, open(output_path, "wb") as f_out:
            make_bert_input_fn = partial(
                make_bert_pretrain_feature,
                max_seq_length=args.max_seq_length,
                short_seq_prob=args.short_seq_prob,
                masked_lm_probs=args.masked_lm_probs,
                max_prediction_per_seq=args.max_prediction_per_seq,
                cls_token_id=tokenizer.token_to_id("[CLS]"),
                sep_token_id=tokenizer.token_to_id("[SEP]"),
                mask_token_id=tokenizer.token_to_id("[MASK]"),
            )
            featuring_iterator = pool.imap_unordered(make_bert_input_fn, zip(documents_chunk, random_documents_chunk))
            for feature in tqdm(featuring_iterator, total=len(documents_chunk), desc="making features"):
                pickle.dump(feature, f_out)

    logger.info("[+] Done!")
    return 0


if __name__ == "__main__":
    exit(main())
