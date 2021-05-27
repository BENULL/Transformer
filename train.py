#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
@author: BENULL
@time: 2021/5/25 15:44
"""
import argparse
import os
import torch
from transformer.Transformer import Transformer
from transformer.Optim import ScheduledOptim
import torch.optim as optim
from tqdm import tqdm
import torch.nn.functional as F
from Dataloader import load_IWSLT
import time
from transformer.Util import create_masks
import math


def cal_performance(pred, gold, trg_pad_idx, smoothing=False):
    """
    Apply label smoothing if needed
    """
    loss = cal_loss(pred, gold, trg_pad_idx, smoothing=smoothing)
    pred = pred.max(1)[1]
    gold = gold.contiguous().view(-1)
    non_pad_mask = gold.ne(trg_pad_idx)
    n_correct = pred.eq(gold).masked_select(non_pad_mask).sum().item()
    n_word = non_pad_mask.sum().item()
    return loss, n_correct, n_word


def cal_loss(pred, gold, trg_pad_idx, smoothing=False):
    """
    Calculate cross entropy loss, apply label smoothing if needed.
    """

    gold = gold.contiguous().view(-1)

    if smoothing:
        eps = 0.1
        n_class = pred.size(1)

        one_hot = torch.zeros_like(pred).scatter(1, gold.view(-1, 1), 1)
        one_hot = one_hot * (1 - eps) + (1 - one_hot) * eps / (n_class - 1)
        log_prb = F.log_softmax(pred, dim=1)

        non_pad_mask = gold.ne(trg_pad_idx)
        loss = -(one_hot * log_prb).sum(dim=1)
        loss = loss.masked_select(non_pad_mask).sum()  # average later
    else:
        loss = F.cross_entropy(pred, gold, ignore_index=trg_pad_idx, reduction='sum')
    return loss


def patch_src(src, pad_idx):
    src = src.transpose(0, 1)
    return src


def patch_trg(trg, pad_idx):
    trg = trg.transpose(0, 1)
    trg, gold = trg[:, :-1], trg[:, 1:].contiguous().view(-1)
    return trg, gold


def train(model, training_data, validation_data, optimizer, device, opt):
    """
    Training
    """
    log_train_file = os.path.join(opt.output_dir, 'train.log')
    log_valid_file = os.path.join(opt.output_dir, 'valid.log')

    print('[Info] Training performance will be written to file: {} and {}'.format(
        log_train_file, log_valid_file))

    with open(log_train_file, 'w') as log_tf, open(log_valid_file, 'w') as log_vf:
        log_tf.write('epoch,loss,ppl,accuracy\n')
        log_vf.write('epoch,loss,ppl,accuracy\n')

    def print_performances(header, ppl, accu, start_time, lr):
        print('  - {header:12} ppl: {ppl: 8.5f}, accuracy: {accu:3.3f} %, lr: {lr:8.5f}, ' \
              'elapse: {elapse:3.3f} min'.format(
            header=f"({header})", ppl=ppl,
            accu=100 * accu, elapse=(time.time() - start_time) / 60, lr=lr))

    valid_losses = []
    for epoch_i in range(opt.epoch):
        print('[ Epoch', epoch_i, ']')

        start = time.time()
        model.train()
        train_loss, train_accu = run_epoch(
            model, training_data, optimizer, opt, device, smoothing=opt.label_smoothing)
        train_ppl = math.exp(min(train_loss, 100))
        # Current learning rate
        lr = optimizer._optimizer.param_groups[0]['lr']
        print_performances('Training', train_ppl, train_accu, start, lr)



        start = time.time()
        model.eval()
        valid_loss, valid_accu = run_epoch(model, validation_data, device, opt, train_epoch=False)
        valid_ppl = math.exp(min(valid_loss, 100))
        print_performances('Validation', valid_ppl, valid_accu, start, lr)

        valid_losses += [valid_loss]

        checkpoint = {'epoch': epoch_i, 'settings': opt, 'model': model.state_dict()}

        if opt.save_mode == 'all':
            model_name = 'model_accu_{accu:3.3f}.chkpt'.format(accu=100 * valid_accu)
            torch.save(checkpoint, model_name)
        elif opt.save_mode == 'best':
            model_name = 'model.chkpt'
            if valid_loss <= min(valid_losses):
                torch.save(checkpoint, os.path.join(opt.output_dir, model_name))
                print('    - [Info] The checkpoint file has been updated.')

        with open(log_train_file, 'a') as log_tf, open(log_valid_file, 'a') as log_vf:
            log_tf.write('{epoch},{loss: 8.5f},{ppl: 8.5f},{accu:3.3f}\n'.format(
                epoch=epoch_i, loss=train_loss,
                ppl=train_ppl, accu=100 * train_accu))
            log_vf.write('{epoch},{loss: 8.5f},{ppl: 8.5f},{accu:3.3f}\n'.format(
                epoch=epoch_i, loss=valid_loss,
                ppl=valid_ppl, accu=100 * valid_accu))

def run_epoch(model, training_data, optimizer, opt, device, smoothing, train_epoch=True):
    """
    Standard Training and Logging Function
    """

    total_loss, n_word_total, n_word_correct = 0, 0, 0

    desc = '  - (Training)   ' if train_epoch else '  - (Testing)   '
    for batch in tqdm(training_data, mininterval=2, desc=desc, leave=False):
        # prepare data
        tar_inp = batch.trg[:, :-1].to(device)
        tar_real = batch.trg[:, 1:].to(device)
        enc_padding_mask, combined_mask, dec_padding_mask = create_masks(batch.src, tar_inp)
        # forward
        optimizer.zero_grad()
        pred = model(batch.src, tar_inp, enc_padding_mask, combined_mask)

        # backward and update parameters
        loss, n_correct, n_word = cal_performance(
            pred, tar_real, opt.trg_pad_idx, smoothing=smoothing)
        loss.backward()
        optimizer.step_and_update_lr()

        # note keeping
        n_word_total += n_word
        n_word_correct += n_correct
        total_loss += loss.item()

    loss_per_word = total_loss / n_word_total
    accuracy = n_word_correct / n_word_total
    return loss_per_word, accuracy


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument('-epoch', type=int, default=10)
    parser.add_argument('-b', '--batch_size', type=int, default=2048)

    parser.add_argument('-d_model', type=int, default=512)
    parser.add_argument('-d_inner_hid', type=int, default=2048)
    parser.add_argument('-d_k', type=int, default=64)
    parser.add_argument('-d_v', type=int, default=64)

    parser.add_argument('-n_head', type=int, default=8)
    parser.add_argument('-n_layer', type=int, default=6)
    parser.add_argument('-warmup', '--n_warmup_steps', type=int, default=4000)
    parser.add_argument('-lr_mul', type=float, default=2.0)

    parser.add_argument('-dropout', type=float, default=0.1)

    parser.add_argument('-output_dir', type=str, default=None)
    parser.add_argument('-save_mode', type=str, choices=['all', 'best'], default='best')
    parser.add_argument('-no_cuda', action='store_true')
    parser.add_argument('-label_smoothing', action='store_true')

    opt = parser.parse_args()

    opt.cuda = not opt.no_cuda

    if not opt.output_dir:
        print('No experiment result will be saved.')
        raise

    if not os.path.exists(opt.output_dir):
        os.makedirs(opt.output_dir)

    if opt.batch_size < 2048 and opt.n_warmup_steps <= 4000:
        print('[Warning] The warmup steps may be not enough.\n'
              '(sz_b, warmup) = (2048, 4000) is the official setting.\n'
              'Using smaller batch w/o longer warmup may cause '
              'the warmup stage ends with only little data trained.')

    device = torch.device('cuda' if opt.cuda else 'cpu')

    # Loading Dataset
    training_data, validation_data = load_IWSLT(opt, device)

    transformer = Transformer(
        src_vocab_size=opt.src_vocab_size,
        tgt_vocab_size=opt.trg_vocab_size,
        n_layer=opt.n_layer,
        d_model=opt.d_model,
        d_ff=opt.d_inner_hid,
        n_head=opt.n_head,
        dropout=opt.dropout).to(device)

    optimizer = ScheduledOptim(
        optim.Adam(transformer.parameters(), betas=(0.9, 0.98), eps=1e-09),
        opt.lr_mul, opt.d_model, opt.n_warmup_steps)

    train(transformer, training_data, validation_data, optimizer, device, opt)


if __name__ == '__main__':
    main()
