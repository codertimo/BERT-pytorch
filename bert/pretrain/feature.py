import random
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple, Union

# input_ids, attention_mask, token_type_ids, position_ids, mlm_targets, nsp_label
BertPretrainFeatures = Dict[str, Union[List[int], List[float], int]]


@dataclass
class Document:
    texts: List[str]
    tokenized_texts: Optional[List[List[int]]] = None


def positive_and_negative_documents_generator(documents: List[Document], num_negative_documents: int = 5):
    for document in documents:
        yield document, random.choices(documents, k=num_negative_documents)


def generate_segment_ab(
    positive_document: Document,
    negative_documents: List[Document],
    max_seq_length: int,
    short_seq_prob: float,
):
    positive_texts_tokens = positive_document.tokenized_texts
    if len(positive_texts_tokens) == 0:
        print(positive_document.texts)
        return

    positive_text_pointer = 0
    while positive_text_pointer < len(positive_texts_tokens):
        segment_a, segment_b = [], []

        target_seq_length = max_seq_length if random.random() > short_seq_prob else random.randint(5, max_seq_length)
        target_seq_length -= 3

        segment_a_target_length = random.randint(1, target_seq_length)
        for i in range(positive_text_pointer, len(positive_texts_tokens)):
            if len(positive_texts_tokens[i]) + len(segment_a) > segment_a_target_length:
                break
            segment_a.extend(positive_texts_tokens[i])
            positive_text_pointer += 1

        segment_b_target_length = target_seq_length - len(segment_a)

        is_nsp_positive = int(random.random() >= 0.5)
        if is_nsp_positive:
            for i in range(positive_text_pointer, len(positive_texts_tokens)):
                positive_text_pointer += 1
                if len(positive_texts_tokens[i]) + len(segment_b) > segment_b_target_length:
                    left_space = segment_b_target_length - len(segment_b)
                    segment_b.extend(positive_texts_tokens[i][:left_space])
                    break
                segment_b.extend(positive_texts_tokens[i])

        else:
            negative_document = random.choice(negative_documents)
            negative_texts_tokens = negative_document.tokenized_texts

            negative_text_start = random.randint(0, len(negative_texts_tokens) - 1)
            for i in range(negative_text_start, len(negative_texts_tokens)):
                if len(negative_texts_tokens[i]) + len(segment_b) > segment_b_target_length:
                    break
                segment_b.extend(negative_texts_tokens[i])

        if segment_a and segment_b:
            yield segment_a, segment_b, is_nsp_positive


def make_features_from_document(
    positive_and_negative_documents: Tuple[Document, List[Document]],
    max_seq_length: int,
    short_seq_prob: float,
    masked_lm_prob: int,
    max_prediction_per_seq: int,
    cls_token_id: int,
    sep_token_id: int,
    mask_token_id: int,
    vocab_size: int,
) -> List[BertPretrainFeatures]:
    features = []
    document, negative_documents = positive_and_negative_documents
    for segment_a, segment_b, is_nsp_positive in generate_segment_ab(
        document, negative_documents, max_seq_length, short_seq_prob
    ):
        # adding special tokens
        segment_a.insert(0, cls_token_id)
        segment_a.append(sep_token_id)
        segment_b.append(sep_token_id)

        input_ids = segment_a + segment_b

        # random masking
        masked_tokens_counts = 0
        masked_lm_targets = []
        for i, token_id in enumerate(input_ids):
            not_special_token = token_id != cls_token_id and token_id != sep_token_id
            is_not_full = masked_tokens_counts < max_prediction_per_seq

            # masked token
            if random.random() < masked_lm_prob and not_special_token and is_not_full:
                # 80% will be mask token
                if random.random() < 0.8:
                    input_ids[i] = mask_token_id

                # 10% will be original token & 10% will be random token
                elif random.random() > 0.5:
                    input_ids[i] = random.randint(3, vocab_size - 1)

                masked_lm_targets.append(token_id)
                masked_tokens_counts += 1

            # non-masked token
            else:
                masked_lm_targets.append(-100)

        # padding and making attention mask and
        attention_mask = [1.0] * len(input_ids)
        token_type_ids = [0] * len(segment_a) + [1] * len(segment_b)
        position_ids = [i for i in range(max_seq_length)]

        padding_size = max_seq_length - len(input_ids)
        if padding_size > 0:
            input_ids.extend([0] * padding_size)
            attention_mask.extend([0.0] * padding_size)
            token_type_ids.extend([0] * padding_size)
            masked_lm_targets.extend([-100] * padding_size)

        assert len(input_ids) == max_seq_length
        assert len(attention_mask) == max_seq_length
        assert len(token_type_ids) == max_seq_length
        assert len(position_ids) == max_seq_length
        assert len(masked_lm_targets) == max_seq_length

        feature = {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "token_type_ids": token_type_ids,
            "position_ids": position_ids,
            "mlm_labels": masked_lm_targets,
            "nsp_label": is_nsp_positive,
        }
        features.append(feature)
    return features
