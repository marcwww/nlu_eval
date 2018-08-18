import preproc
import nets
from macros import *
import torch
import utils
import opts
import argparse
import training
from torch import nn
from torch import optim

if __name__ == '__main__':
    parser = argparse. \
        ArgumentParser(description='main.py',
                       formatter_class=argparse.
                       ArgumentDefaultsHelpFormatter)

    opts.model_opts(parser)
    opts.train_opts(parser)
    opt = parser.parse_args()

    train_iter, valid_iter, SEQ, LBL = \
                        preproc.build_iters(ftrain=opt.ftrain,
                        fvalid=opt.fvalid,
                        bsz=opt.bsz,
                        min_freq=opt.min_freq,
                        device=opt.gpu)

    encoder = nets.EncoderSRNN(voc_size=len(SEQ.vocab.itos),
                               edim=opt.edim,
                               hdim=opt.hdim,
                               stack_size=opt.stack_size,
                               sdim=opt.sdim,
                               padding_idx=SEQ.vocab.stoi[PAD],
                               stack_depth=opt.stack_depth)

    model = nets.TextualEntailmentModel(encoder=encoder,
                                        nclasses=len(LBL.vocab.itos))

    utils.init_model(model)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(params=filter(lambda p: p.requires_grad, model.parameters()),
        lr=opt.lr, weight_decay=opt.wdecay)

    training.train_rte(model,
                       {'train': train_iter,
                        'valid':valid_iter},
                       opt,
                       criterion,
                       optimizer)


