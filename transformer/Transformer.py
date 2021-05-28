#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
@author: BENULL
@time: 2021/5/22 15:22
"""
import torch.nn as nn
from transformer.Embedding import Embeddings, PositionalEncoding
from transformer.Models import Encoder, Decoder, Generator


class Transformer(nn.Module):
    """
    A sequence to sequence model with attention mechanism.
    """

    def __init__(self, src_vocab_size, tgt_vocab_size, n_layer=6,
                 d_model=512, d_ff=2048, n_head=8, dropout=0.1):
        super(Transformer, self).__init__()
        self.src_embed = nn.Sequential(Embeddings(d_model, src_vocab_size), PositionalEncoding(d_model, dropout))
        self.tgt_embed = nn.Sequential(Embeddings(d_model, tgt_vocab_size), PositionalEncoding(d_model, dropout))
        self.encoder = Encoder(n_head, d_model, d_ff, dropout, n_layer)
        self.decoder = Decoder(n_head, d_model, d_ff, dropout, n_layer)
        self.generator = Generator(d_model, tgt_vocab_size)

        # Initialize parameters with Glorot / fan_avg.
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, src, tgt, src_mask, tgt_mask):
        """
        Take in and process masked src and target sequences.
        """
        x = self.decode(self.encode(src, src_mask), src_mask,
                           tgt, tgt_mask)
        # x = self.generator(x)
        return x

    def encode(self, src, src_mask):
        return self.encoder(self.src_embed(src), src_mask)

    def decode(self, enc_output, src_mask, tgt, tgt_mask):
        return self.decoder(self.tgt_embed(tgt), enc_output, src_mask, tgt_mask)

