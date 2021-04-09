import pickle
from typing import List

from .feature import BertPretrainFeatures, Document


def load_corpus(corpus_path: str) -> List[Document]:
    documents, texts = [], []
    with open(corpus_path) as f:
        for line in f:
            line = line.strip()
            if line:
                texts.append(line)
            elif len(texts) > 0 and sum(map(len, texts)) > 64:
                documents.append(Document(texts))
                texts = []
            else:
                texts = []
    return documents


def load_feature_records(record_file_path: str) -> List[BertPretrainFeatures]:
    with open(record_file_path, "rb") as f:
        feature_list = []
        while True:
            try:
                data = pickle.load(f)
            except EOFError:
                break
            feature_list.append(data)
        return feature_list


def save_feature_records(record_file_path: str, features: List[BertPretrainFeatures]):
    with open(record_file_path, "wb") as f:
        for feature in features:
            pickle.dump(feature, f)
