#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
@author: BENULL
@time: 2021/5/22 15:26
"""
import torch.nn as nn
from transformer.Util import clone
from transformer.Sublayers import MultiHeadedAttention, PositionwiseFeedForward, LayerNorm


class DecoderLayer(nn.Module):
    """
    Decoder is made of self-attn, src-attn, and feed forward
    """

    def __init__(self, h, d_model, d_ff, dropout):
        super(DecoderLayer, self).__init__()

        self.self_attn = MultiHeadedAttention(h, d_model)
        self.src_attn = MultiHeadedAttention(h, d_model)
        self.ffn = PositionwiseFeedForward(d_model, d_ff, dropout)
        self.sublayer = clone(SublayerConnection(d_model, dropout), 3)

    def forward(self, x, enc_output, look_ahead_mask, padding_mask):
        enc_output = enc_output
        x = self.sublayer[0](x, lambda x: self.self_attn(x, x, x, padding_mask))
        x = self.sublayer[1](x, lambda x: self.src_attn(x, enc_output, enc_output, look_ahead_mask))
        return self.sublayer[2](x, self.ffn)


class EncoderLayer(nn.Module):
    """
    Encoder is made up of self-attention and feed forward
    """

    def __init__(self, h, d_model, d_ff, dropout):
        super(EncoderLayer, self).__init__()
        self.mha = MultiHeadedAttention(h, d_model)
        self.ffn = PositionwiseFeedForward(d_model, d_ff, dropout)
        self.sublayer = clone(SublayerConnection(d_model, dropout), 2)

    def forward(self, x, mask):
        x = self.sublayer[0](x, lambda x: self.mha(x, x, x, mask))
        return self.sublayer[1](x, self.ffn)


class SublayerConnection(nn.Module):
    """
    A residual connection followed by a layer norm.
    """

    def __init__(self, normalized_shape, dropout):
        super(SublayerConnection, self).__init__()
        self.norm = LayerNorm(normalized_shape)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, sublayer):
        return x + self.dropout(sublayer(self.norm(x)))
