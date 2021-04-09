import os
import random
from argparse import ArgumentParser, Namespace
from functools import partial
from glob import glob
from multiprocessing import Pool

import nltk
from tokenizers import BertWordPieceTokenizer, Tokenizer
from tqdm import tqdm

from bert.pretrain.feature import make_features_from_document, positive_and_negative_documents_generator
from bert.pretrain.utils import load_corpus, save_feature_records
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
parser.add_argument("--num-workers", default=10, type=int, help="num worker to multi-process")
parser.add_argument("--num-features-per-file", default=100000, type=int, help="number of features to save on single file")
parser.add_argument("--vocab-size", default=50004, type=int, help="vocab size of tokenizer")
parser.add_argument("--use-sentence-splitter", action="store_true", help="split line using nltk splitter")
parser.add_argument("--pretrained-tokenizer-json-path", type=str, help="(optional) pretrained tokenizer json path, don't give anything to pretrained-tokenizer-vocab-{path, is-lowercase}")
parser.add_argument("--pretrained-tokenizer-vocab-path", type=str, help="(optional) pretrained tokenizer vocab path")
parser.add_argument("--pretrained-tokenizer-is-uncased", action="store_true", help="(optional, default:False) if pretrained-tokenizer-vocab is not None")
parser.add_argument("--random-seed", default=0, type=int, help="random seed")
# fmt: on

logger = get_logger("BERT-data")
os.environ["TOKENIZERS_PARALLELISM"] = "true"


def main(args: Namespace) -> int:
    random.seed(args.random_seed)
    os.makedirs(args.output_dir, exist_ok=True)

    documents = []
    for corpus_path in glob(args.corpus_paths):
        logger.info(f"[+] Parsing corpus: {corpus_path}")
        documents.extend(load_corpus(corpus_path))

    if args.use_sentence_splitter:
        nltk.download("punkt")
        logger.info("[+] Splitting long text line into sentences")
        for document in documents:
            document.texts = [splited_text for text in document.texts for splited_text in nltk.sent_tokenize(text)]

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
        tokenizer.train(glob(args.corpus_paths), vocab_size=args.vocab_size, min_frequency=1)

        trained_tokenizer_path = os.path.join(args.output_dir, "tokenizer.json")
        logger.info(f"[+] Saving trained WordPieceTokeizer to {trained_tokenizer_path}")
        tokenizer.save(trained_tokenizer_path, pretty=True)

    if tokenizer.get_vocab_size() != args.vocab_size:
        logger.warning(f"[-] arg.vocab_size({args.vocab_size}) != tokenizer.vocab_size({tokenizer.get_vocab_size()})")

    num_total_texts = sum([len(document.texts) for document in documents])
    logger.info(f"[+] Tokenizing {len(documents)} documents {num_total_texts} texts start")
    tokenizing_candidates = [text for document in documents for text in document.texts]
    tokenized_texts = tokenizer.encode_batch(tokenizing_candidates, add_special_tokens=False)
    for document in documents:
        document.tokenized_texts = [tokenized_texts.pop(0).ids for _ in range(len(document.texts))]

    documents = documents * args.num_duplicates
    logger.info(f"[+] Making BERT pre-training input features using {len(documents)} documents")

    def feature_generator():
        with Pool(args.num_workers) as pool:
            feature_maker_fn = partial(
                make_features_from_document,
                max_seq_length=args.max_seq_length,
                short_seq_prob=args.short_seq_prob,
                masked_lm_prob=args.masked_lm_prob,
                max_prediction_per_seq=args.max_prediction_per_seq,
                cls_token_id=tokenizer.token_to_id("[CLS]"),
                sep_token_id=tokenizer.token_to_id("[SEP]"),
                mask_token_id=tokenizer.token_to_id("[MASK]"),
                vocab_size=tokenizer.get_vocab_size(),
            )
            feature_iter = pool.imap_unordered(feature_maker_fn, positive_and_negative_documents_generator(documents))
            for features in tqdm(feature_iter, total=len(documents), desc="feature generating"):
                for feature in features:
                    yield feature

    feature_buffer, record_file_index = [], 0
    logger.info(f"[+] Generating {len(documents)} features")
    for feature_id, feature in enumerate(feature_generator()):
        feature_buffer.append(feature)

        if len(feature_buffer) == args.num_features_per_file:
            output_path = os.path.join(args.output_dir, f"bert_pretrain_feature.{record_file_index:06d}.records")
            logger.info(f"[+] Wrting {len(feature_buffer)} features to {output_path}")
            save_feature_records(output_path, feature_buffer)
            feature_buffer.clear()
            record_file_index += 1

        if feature_id < 5:
            logger.info(f"======feature-{feature_id}======")
            for key, value in feature.items():
                logger.info(f"{key}: {value}")
            logger.info("\n")

    output_path = os.path.join(args.output_dir, f"bert_pretrain_feature.{record_file_index:06d}.records")
    logger.info(f"[+] Wrting {len(feature_buffer)} features to {output_path}")
    save_feature_records(output_path, feature_buffer)

    logger.info(f"[+] total {feature_id+1} features wrote into {record_file_index+1} record files")
    logger.info("[+] Done!")
    return 0


if __name__ == "__main__":
    exit(main(parser.parse_args()))
