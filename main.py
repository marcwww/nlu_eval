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
from tasks import prop_entailment


if __name__ == '__main__':
    parser = argparse. \
        ArgumentParser(description='main.py',
                       formatter_class=argparse.
                       ArgumentDefaultsHelpFormatter)

    opts.model_opts(parser)
    opts.train_opts(parser)
    opt = parser.parse_args()

    build_iters = None
    train = None
    Model = None
    criterion = None
    if opt.task == 'prop-entail':
        build_iters = prop_entailment.build_iters
        train = prop_entailment.train
        Model = prop_entailment.Model

    res_iters = build_iters(ftrain=opt.ftrain,
                fvalid=opt.fvalid,
                bsz=opt.bsz,
                device=opt.gpu)

    train_iter = res_iters['train_iter']
    valid_iter = res_iters['valid_iter']
    SEQ = res_iters['SEQ']
    LBL = res_iters['LBL']

    embedding = nn.Embedding(num_embeddings=len(SEQ.vocab.itos),
                 embedding_dim=opt.edim,
                 padding_idx=SEQ.vocab.stoi[PAD])

    location = opt.gpu if torch.cuda.is_available() and opt.gpu != -1 else 'cpu'
    device = torch.device(location)

    if opt.emb_type == 'one-hot':
        one_hot_mtrx = utils.one_hot_matrix(SEQ.vocab.stoi, device, opt.edim)
        embedding.weight.data.copy_(one_hot_mtrx)

    encoder = nets.EncoderSimpRNN(idim=opt.edim,
                                  hdim=opt.hdim)

    model = Model(encoder, embedding).to(device)
    utils.init_model(model)

    if opt.load_idx != -1:
        basename = "{}-epoch-{}".format(opt.task, opt.load_idx)
        model_fname = basename + ".model"
        location = {'cuda:' + str(opt.gpu): 'cuda:' + str(opt.gpu)} if opt.gpu != -1 else 'cpu'
        model_path = os.path.join(RES, model_fname)
        model_dict = torch.load(model_path, map_location=location)
        model.load_state_dict(model_dict)
        print('Loaded from ' + model_path)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(params=filter(lambda p: p.requires_grad, model.parameters()),
                           lr=opt.lr,
                           weight_decay=opt.wdecay)

    iters = {'train': train_iter, 'valid': valid_iter}
    train(model, iters, opt, criterion, optimizer)

