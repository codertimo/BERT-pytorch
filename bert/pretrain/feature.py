import random
from dataclasses import dataclass
from typing import List, Optional, Tuple

# input_ids, attention_mask, token_type_ids, position_ids, mlm_targets, nsp_label
BertPretrainFeature = Tuple[List[int], List[float], List[int], List[int], List[int], int]


@dataclass
class Document:
    texts: List[str]
    tokenized_texts: Optional[List[List[int]]] = None


def make_bert_pretrain_feature(
    positive_and_negative_documents: Tuple[Document, Document],
    max_seq_length: int,
    short_seq_prob: float,
    masked_lm_prob: int,
    max_prediction_per_seq: int,
    cls_token_id: int,
    sep_token_id: int,
    mask_token_id: int,
) -> BertPretrainFeature:
    positive_document, negative_document = positive_and_negative_documents

    # reserve special token space
    target_seq_length = max_seq_length - 3
    if random.random() < short_seq_prob:
        target_seq_length = random.randint(2, max_seq_length)

    segment_a, segment_b = [], []
    segment_a_target_length = random.randint(1, target_seq_length - 1)

    # postiive next sentence prediction sample
    is_nsp_positive = int(random.random() >= 0.5)
    if is_nsp_positive:
        for tokenized_text in positive_document.tokenized_texts:
            if len(segment_a) + len(tokenized_text) <= segment_a_target_length:
                segment_a.extend(tokenized_text)
            elif len(segment_a) + len(segment_b) + len(tokenized_text) <= target_seq_length:
                segment_b.extend(tokenized_text)
            else:
                break
    # negative next sentence prediction sample
    else:
        for tokenized_text in positive_document.tokenized_texts:
            if len(segment_a) + len(tokenized_text) > segment_a_target_length:
                break
            segment_a.extend(tokenized_text)

        for tokenized_text in negative_document.tokenized_texts:
            if len(segment_a) + len(segment_b) + len(tokenized_text) > target_seq_length:
                break
            segment_b.extend(tokenized_text)

    # adding special tokens
    segment_a.insert(0, cls_token_id)
    segment_a.append(sep_token_id)
    segment_b.append(sep_token_id)
    input_ids = segment_a + segment_b

    # random masking
    masked_tokens_counts = 0
    masked_lm_targets = [], []
    for i, token_id in enumerate(input_ids):
        not_special_token = token_id != cls_token_id and token_id != sep_token_id
        if not_special_token and masked_tokens_counts < max_prediction_per_seq and random.random() > masked_lm_prob:
            input_ids[i] = mask_token_id
            masked_lm_targets.append(token_id)
            masked_tokens_counts += 1
        else:
            masked_lm_targets.append(-100)

    # padding and making attention mask and
    padding_size = max_seq_length - len(input_ids)
    attention_mask = [1.0] * len(input_ids) + [0.0] * padding_size
    input_ids.extend([0] * len(padding_size))
    token_type_ids = [0] * len(segment_a) + [1] * len(segment_b)
    position_ids = [i for i in range(max_seq_length)]
    masked_lm_targets.extend([-100] * padding_size)

    assert len(input_ids) == max_seq_length
    assert len(attention_mask) == max_seq_length
    assert len(token_type_ids) == max_seq_length
    assert len(position_ids) == max_seq_length
    assert len(masked_lm_targets) == max_seq_length

    return input_ids, attention_mask, token_type_ids, position_ids, masked_lm_targets, is_nsp_positive


def tokenize_document(document: Document, tokenizer):
    document.tokenized_texts = [tokenizer.tokenize(text) for text in document.texts]


def load_corpus(corpus_path: str) -> List[Document]:
    documents, texts = [], []
    with open(load_corpus) as f:
        for line in f:
            line = line.strip()
            if line:
                texts.append(line)
            elif len(texts) > 0:
                documents.append(Document(texts))
                texts = []
    return documents
