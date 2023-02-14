# BERT-pytorch

[![LICENSE](https://img.shields.io/github/license/codertimo/BERT-pytorch.svg)](https://github.com/codertimo/BERT-pytorch/blob/master/LICENSE)
![GitHub issues](https://img.shields.io/github/issues/codertimo/BERT-pytorch.svg)
[![GitHub stars](https://img.shields.io/github/stars/codertimo/BERT-pytorch.svg)](https://github.com/codertimo/BERT-pytorch/stargazers)
[![CircleCI](https://circleci.com/gh/codertimo/BERT-pytorch.svg?style=shield)](https://circleci.com/gh/codertimo/BERT-pytorch)
[![PyPI](https://img.shields.io/pypi/v/bert-pytorch.svg)](https://pypi.org/project/bert_pytorch/)
[![PyPI - Status](https://img.shields.io/pypi/status/bert-pytorch.svg)](https://pypi.org/project/bert_pytorch/)
[![Documentation Status](https://readthedocs.org/projects/bert-pytorch/badge/?version=latest)](https://bert-pytorch.readthedocs.io/en/latest/?badge=latest)

Pytorch implementation of Google AI's 2018 BERT, with simple annotation

> BERT 2018 BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding
> Paper URL : https://arxiv.org/abs/1810.04805


## Introduction

Google AI's BERT paper shows the amazing result on various NLP task (new 17 NLP tasks SOTA), 
including outperform the human F1 score on SQuAD v1.1 QA task. 
This paper proved that Transformer(self-attention) based encoder can be powerfully used as 
alternative of previous language model with proper language model training method. 
And more importantly, they showed us that this pre-trained language model can be transfer 
into any NLP task without making task specific model architecture.

This amazing result would be record in NLP history, 
and I expect many further papers about BERT will be published very soon.

This repo is implementation of BERT. Code is very simple and easy to understand fastly.
Some of these codes are based on [The Annotated Transformer](http://nlp.seas.harvard.edu/2018/04/03/attention.html)

Currently this project is working on progress. And the code is not verified yet.

## Installation
```
pip install bert-pytorch
```

## Quickstart

**NOTICE : Your corpus should be prepared with two sentences in one line with tab(\t) separator**

### 0. Prepare your corpus
```
Welcome to the \t the jungle\n
I can stay \t here all night\n
```

or tokenized corpus (tokenization is not in package)
```
Wel_ _come _to _the \t _the _jungle\n
_I _can _stay \t _here _all _night\n
```


### 1. Building vocab based on your corpus
```shell
bert-vocab -c data/corpus.small -o data/vocab.small
```

### 2. Train your own BERT model
```shell
bert -c data/corpus.small -v data/vocab.small -o output/bert.model
```

## Language Model Pre-training

In the paper, authors shows the new language model training methods, 
which are "masked language model" and "predict next sentence".


### Masked Language Model 

> Original Paper : 3.3.1 Task #1: Masked LM 

```
Input Sequence  : The man went to [MASK] store with [MASK] dog
Target Sequence :                  the                his
```

#### Rules:
Randomly 15% of input token will be changed into something, based on under sub-rules

1. Randomly 80% of tokens, gonna be a `[MASK]` token
2. Randomly 10% of tokens, gonna be a `[RANDOM]` token(another word)
3. Randomly 10% of tokens, will be remain as same. But need to be predicted.

### Predict Next Sentence

> Original Paper : 3.3.2 Task #2: Next Sentence Prediction

```
Input : [CLS] the man went to the store [SEP] he bought a gallon of milk [SEP]
Label : Is Next

Input = [CLS] the man heading to the store [SEP] penguin [MASK] are flight ##less birds [SEP]
Label = NotNext
```

"Is this sentence can be continuously connected?"

 understanding the relationship, between two text sentences, which is
not directly captured by language modeling

#### Rules:

1. Randomly 50% of next sentence, gonna be continuous sentence.
2. Randomly 50% of next sentence, gonna be unrelated sentence.


## Author
Junseong Kim, Scatter Lab (codertimo@gmail.com / junseong.kim@scatterlab.co.kr)

## License

This project following Apache 2.0 License as written in LICENSE file

Copyright 2018 Junseong Kim, Scatter Lab, respective BERT contributors

Copyright (c) 2018 Alexander Rush : [The Annotated Trasnformer](https://github.com/harvardnlp/annotated-transformer)
