import torchtext
from macros import *
import os
import crash_on_ipy

def build_iters(ftrain, fvalid, bsz, device, min_freq, pretrain):

    IDX = torchtext.data.Field(sequential=False, use_vocab=False)
    SEQ = torchtext.data.Field(sequential=True, use_vocab=True,
                               pad_token=PAD,
                               unk_token=UNK,
                               eos_token=EOS)
    LBL = torchtext.data.Field(sequential=False, use_vocab=True)
    # LBL = torchtext.data.Field(sequential=True, use_vocab=True, eos_token=None, pad_token=None, unk_token=None)
    # with open(ftrain, 'r') as f:
    #     lines = f.readlines()[0:]
    #     for line in lines:
    #         print(line.split('\t'))

    train = torchtext.data.TabularDataset(path=ftrain,
                                          format='tsv',
                                          skip_header=True,
                                          fields=[
                                              ('idx', IDX),
                                              ('seq1', SEQ),
                                              ('seq2', SEQ),
                                              ('lbl', LBL)])

    SEQ.build_vocab(train, min_freq=min_freq, vectors=pretrain)
    LBL.build_vocab(train)

    valid = torchtext.data.TabularDataset(path=fvalid,
                                          format='tsv',
                                          skip_header=True,
                                          fields=[
                                              ('idx', IDX),
                                              ('seq1', SEQ),
                                              ('seq2', SEQ),
                                              ('lbl', LBL)])

    train_iter = torchtext.data.Iterator(train, batch_size=bsz,
                                         sort=False, repeat=False,
                                         sort_key=lambda x: len(x.seq1),
                                         sort_within_batch=True,
                                         device=device)
    valid_iter = torchtext.data.Iterator(valid, batch_size=bsz,
                                         sort=False, repeat=False,
                                         sort_key=lambda x: len(x.seq1),
                                         sort_within_batch=True,
                                         device=device)

    return train_iter, valid_iter, SEQ, LBL

