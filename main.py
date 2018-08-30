import nets
from macros import *
import torch
import utils
import opts
import argparse
import training
from torch import nn
from torch import optim
from tasks import prop_entailment, \
    prop_entailment_2enc, \
    rewriting, \
    rte, \
    scan, \
    pattern


if __name__ == '__main__':
    parser = argparse. \
        ArgumentParser(description='main.py',
                       formatter_class=argparse.
                       ArgumentDefaultsHelpFormatter)

    opts.model_opts(parser)
    opts.train_opts(parser)
    opt = parser.parse_args()

    utils.init_seed(opt.seed)

    build_iters = None
    train = None
    Model = None
    criterion = None
    if opt.task == 'prop-entail':
        build_iters = prop_entailment.build_iters
        train = prop_entailment.train
        Model = prop_entailment.Model

    if opt.task == 'prop-entail-2enc':
        build_iters = prop_entailment_2enc.build_iters
        train = prop_entailment_2enc.train
        Model = prop_entailment_2enc.Model

    if opt.task == 'rewriting':
        build_iters = rewriting.build_iters
        train = rewriting.train
        Model = rewriting.Model

    if opt.task == 'rte':
        build_iters = rte.build_iters
        train = rte.train
        Model = rte.Model

    if opt.task == 'scan':
        build_iters = scan.build_iters
        train = scan.train
        Model = scan.Model

    if opt.task == 'pattern':
        build_iters = pattern.build_iters
        train = pattern.train
        Model = pattern.Model

    param_iter = {'ftrain': opt.ftrain,
                  'fvalid': opt.fvalid,
                  'bsz': opt.bsz,
                  'device': opt.gpu,
                  'sub_task': opt.sub_task,
                  'num_batches_train': opt.num_batches_train,
                  'num_batches_valid': opt.num_batches_valid,
                  'min_len_train': opt.min_len_train,
                  'min_len_valid': opt.min_len_valid,
                  'max_len_train': opt.max_len_train,
                  'max_len_valid': opt.max_len_valid,
                  'repeat_min_train': opt.repeat_min_train,
                  'repeat_max_train': opt.repeat_max_train,
                  'repeat_min_valid': opt.repeat_min_valid,
                  'repeat_max_valid': opt.repeat_max_valid,
                  'seq_width': opt.seq_width}

    res_iters = build_iters(param_iter)

    embedding = None
    embedding_enc = None
    embedding_dec = None
    SEQ = None
    SRC = None
    TAR = None
    if 'SEQ' in res_iters.keys():
        SEQ = res_iters['SEQ']
        embedding = nn.Embedding(num_embeddings=len(SEQ.vocab.itos),
                                 embedding_dim=opt.edim,
                                 padding_idx=SEQ.vocab.stoi[PAD])

    if 'SRC' in res_iters.keys() and 'TAR' in res_iters.keys():
        SRC = res_iters['SRC']
        embedding_enc = nn.Embedding(num_embeddings=len(SRC.vocab.itos),
                                     embedding_dim=opt.edim,
                                     padding_idx=SRC.vocab.stoi[PAD])

        TAR = res_iters['TAR']
        embedding_dec = nn.Embedding(num_embeddings=len(TAR.vocab.itos),
                                 embedding_dim=opt.edim,
                                 padding_idx=TAR.vocab.stoi[PAD])

    location = opt.gpu if torch.cuda.is_available() and opt.gpu != -1 else 'cpu'
    device = torch.device(location)

    if opt.emb_type == 'one-hot':
        if embedding is not None:
            one_hot_mtrx = utils.one_hot_matrix(SEQ.vocab.stoi, device, opt.edim)
            embedding.weight.data.copy_(one_hot_mtrx)
            embedding.weight.requires_grad = False

        if embedding_enc is not None:
            one_hot_mtrx = utils.one_hot_matrix(SRC.vocab.stoi, device, opt.edim)
            embedding_enc.weight.data.copy_(one_hot_mtrx)
            embedding_enc.weight.requires_grad = False

        if embedding_dec is not None:
            one_hot_mtrx = utils.one_hot_matrix(TAR.vocab.stoi, device, opt.edim)
            embedding_dec.weight.data.copy_(one_hot_mtrx)
            embedding_dec.weight.requires_grad = False

    encoder = None
    decoder = None
    if opt.enc_type == 'simp-rnn':
        encoder = nets.EncoderSimpRNN(idim=opt.edim,
                                      hdim=opt.hdim,
                                      dropout=opt.dropout)

    if opt.enc_type == 'lstm':
        encoder = nets.EncoderLSTM(idim=opt.edim,
                                   hdim=opt.hdim,
                                   dropout=opt.dropout)

    if opt.enc_type == 'ntm':
        encoder = nets.EncoderNTM(idim=opt.edim,
                                  cdim=opt.hdim,
                                  num_heads=opt.num_heads,
                                  N=opt.N,
                                  M=opt.M)
    if opt.enc_type == 'dnc':
        encoder = nets.EncoderDNC(idim=opt.edim,
                                  cdim=opt.hdim,
                                  num_heads=opt.num_heads,
                                  N=opt.N,
                                  M=opt.M,
                                  gpu=opt.gpu)
    if opt.enc_type == 'sarnn':
        encoder = nets.EncoderSARNN(idim=opt.edim,
                                    hdim=opt.hdim,
                                    nstack=opt.nstack,
                                    stack_size=opt.stack_size,
                                    sdim=opt.sdim,
                                    sdepth=opt.stack_depth)
    if opt.enc_type == 'nse':
        encoder = nets.EncoderNSE(idim=opt.edim,
                                  dropout=opt.dropout)

    if opt.enc_type == 'tardis':
        a = int(opt.M * opt.a_ratio)
        c = opt.M - a
        encoder = nets.EncoderTARDIS(idim=opt.edim,
                                     hdim=opt.hdim,
                                     N=opt.N,
                                     a=a,
                                     c=c)

    if opt.dec_type == 'simp-rnn':
        decoder = nets.DecoderSimpRNN(idim=opt.edim,
                                      hdim=opt.hdim,
                                      dropout=opt.dropout)

    if opt.dec_type == 'lstm':
        decoder = nets.DecoderLSTM(idim=opt.edim,
                                   hdim=opt.hdim,
                                   dropout=opt.dropout)

    if opt.dec_type == 'alstm':
        decoder = nets.DecoderALSTM(idim=opt.edim,
                                    hdim=opt.hdim,
                                    dropout=opt.dropout)

    if opt.dec_type == 'ntm':
        decoder = nets.DecoderNTM(idim=opt.edim,
                                  cdim=opt.hdim,
                                  num_heads=opt.num_heads,
                                  N=opt.N,
                                  M=opt.M)
    if opt.dec_type == 'nse':
        decoder = nets.DecoderNSE(idim=opt.edim,
                                  N=opt.N,
                                  dropout=opt.dropout)

    model = None
    if TAR is None:
        if embedding is None:
            model = Model(encoder, opt.odim).to(device)
        else:
            model = Model(encoder, embedding).to(device)
        utils.init_model(model)
    else:
        model = Model(encoder, decoder,
                      embedding_enc, embedding_dec,
                      TAR.vocab.stoi[SOS]).to(device)
        # utils.init_model(model)

    if opt.load_idx != -1:
        basename = "{}-epoch-{}".format(opt.task, opt.load_idx)
        model_fname = basename + ".model"
        location = {'cuda:' + str(opt.gpu): 'cuda:' + str(opt.gpu)} if opt.gpu != -1 else 'cpu'
        model_path = os.path.join(RES, model_fname)
        model_dict = torch.load(model_path, map_location=location)
        model.load_state_dict(model_dict)
        print('Loaded from ' + model_path)

    if TAR is None:
        criterion = nn.CrossEntropyLoss()
    else:
        criterion = nn.CrossEntropyLoss(ignore_index=TAR.vocab.stoi[PAD])

    if opt.task == 'pattern':
        criterion = nn.BCELoss()

    optimizer = optim.Adam(params=filter(lambda p: p.requires_grad, model.parameters()),
                           lr=opt.lr,
                           weight_decay=opt.wdecay)
    # optimizer = optim.SGD(params=filter(lambda p: p.requires_grad, model.parameters()),
    #                        lr=opt.lr,
    #                        weight_decay=opt.wdecay)
    # optimizer = optim.RMSprop(params=filter(lambda p: p.requires_grad, model.parameters()),
    #                           momentum=0.9,
    #                           alpha=0.95,
    #                           lr=1e-4)

    train(model, res_iters, opt, criterion, optimizer)

