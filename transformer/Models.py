#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
@author: BENULL
@time: 2021/5/22 15:26
"""
import torch.nn as nn
import torch.nn.functional as F
from transformer.util import clone
from transformer.Layers import EncoderLayer, DecoderLayer
from transformer.Sublayers import LayerNorm


class Encoder(nn.Module):
    """
    Core encoder is a stack of N layers
    """

    def __init__(self, h, d_model, d_ff, dropout, N):
        super(Encoder, self).__init__()
        self.layers = clone(EncoderLayer(h, d_model, d_ff, dropout), N)
        self.norm = LayerNorm(d_model)

    def forward(self, x, mask):
        # Pass the input (and mask) through each layer in turn.
        for layer in self.layers:
            x = layer(x, mask)
        return self.norm(x)


class Decoder(nn.Module):
    """
    Decoder is a stack of N layers with masking.
    """

    def __init__(self, h, d_model, d_ff, dropout, N):
        super(Decoder, self).__init__()
        self.layers = clone(DecoderLayer(h, d_model, d_ff, dropout), N)
        self.norm = LayerNorm(d_model)

    def forward(self, x, enc_output, src_mask, tgt_mask):
        for layer in self.layers:
            x = layer(x, enc_output, src_mask, tgt_mask)
        return self.norm(x)


class Generator(nn.Module):
    """
    Define standard linear + softmax generation step.
    """

    def __init__(self, d_model, vocab):
        super(Generator, self).__init__()
        self.proj = nn.Linear(d_model, vocab)

    def forward(self, x):
        return F.log_softmax(self.proj(x), dim=-1)
