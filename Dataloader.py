#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
@author: BENULL
@time: 2021/5/25 19:42
"""

from torchtext import data, datasets
import spacy
from torchtext.data import Field, Dataset, BucketIterator
from torchtext.datasets import TranslationDataset

spacy_de = spacy.load('de')
spacy_en = spacy.load('en')

def tokenize_de(text):
    return [tok.text for tok in spacy_de.tokenizer(text)]


def tokenize_en(text):
    return [tok.text for tok in spacy_en.tokenizer(text)]


def load_IWSLT(opt, device):
    BOS_WORD = '<s>'
    EOS_WORD = '</s>'
    BLANK_WORD = "<blank>"
    SRC = data.Field(tokenize=tokenize_de, pad_token=BLANK_WORD)
    TGT = data.Field(tokenize=tokenize_en, init_token=BOS_WORD,
                     eos_token=EOS_WORD, pad_token=BLANK_WORD)

    MAX_LEN = 100
    train, val, test = datasets.IWSLT.splits(
        exts=('.de', '.en'), fields=(SRC, TGT),
        filter_pred=lambda x: len(vars(x)['src']) <= MAX_LEN and
                              len(vars(x)['trg']) <= MAX_LEN)
    MIN_FREQ = 2
    SRC.build_vocab(train.src, min_freq=MIN_FREQ)
    TGT.build_vocab(train.trg, min_freq=MIN_FREQ)

    opt.max_token_seq_len = MAX_LEN
    opt.src_pad_idx = SRC.vocab.stoi[BLANK_WORD]
    opt.trg_pad_idx = TGT.vocab.stoi[BLANK_WORD]
    opt.src_vocab_size = len(SRC.vocab)
    opt.trg_vocab_size = len(TGT.vocab)

    fields = {'src':SRC, 'trg':TGT}

    train = Dataset(examples=train.examples, fields=fields)
    val = Dataset(examples=val.examples, fields=fields)

    train_iterator = BucketIterator(train, batch_size=opt.batch_size, device=device, train=True)
    val_iterator = BucketIterator(val, batch_size=opt.batch_size, device=device)


    return train_iterator, val_iterator
