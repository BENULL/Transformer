#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
@author: BENULL
@time: 2021/5/22 15:32
"""
import torch
import torch.nn as nn
import copy
from torch.autograd import Variable
import numpy as np

def subsequent_mask(size):
    """
    Mask out subsequent positions
    """
    attn_shape = (1, size, size)
    subsequent_mask = np.triu(np.ones(attn_shape), k=1).astype('uint8')
    return torch.from_numpy(subsequent_mask) == 0

    # look_ahead_mask = (1 - torch.triu(
    #     torch.ones((1, size, size)), diagonal=1)).bool()
    # return look_ahead_mask  # (size, size)


def get_pad_mask(seq, pad_idx=0):
    return (seq != pad_idx).unsqueeze(-2)


def create_masks(src, tar):
    # Encoder padding mask
    enc_padding_mask = get_pad_mask(src)

    # Used in the 2nd attention block in the decoder.
    # This padding mask is used to mask the encoder outputs.
    dec_padding_mask = get_pad_mask(src)

    # Used in the 1st attention block in the decoder.
    # It is used to pad and mask future tokens in the input received by the decoder.
    look_ahead_mask = subsequent_mask(tar.size(-1))
    dec_target_padding_mask = get_pad_mask(tar)

    combined_mask = dec_target_padding_mask & Variable(
        look_ahead_mask.type_as(dec_target_padding_mask.data))

    return enc_padding_mask, combined_mask, dec_padding_mask


def clone(module, N):
    """
    Produce N identical layers.
    """
    return nn.ModuleList([copy.deepcopy(module) for _ in range(N)])
