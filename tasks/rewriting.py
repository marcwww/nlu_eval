import torchtext
from macros import *
from torchtext.data import Dataset
import torch
from torch import nn
from torch.nn.utils import clip_grad_norm
import numpy as np
import utils
import itertools
from collections import defaultdict
from sklearn.metrics import accuracy_score, \
    precision_score, recall_score, f1_score

class Example(object):

    def __init__(self, src, tar):
        self.src = self.tokenizer(src)[:-1]
        self.tar = self.tokenizer(tar)

    def tokenizer(self, seq):
        res = seq.split()
        return res

def load_examples(fname):
    examples = []

    with open(fname, 'r') as f:
        for line in f:
            src, tar, _ = line.split('\t')
            examples.append(Example(src, tar))

    return examples

def alpha_order(ch):
    return ord(ch) - ord('A') + 1

def record_map(src, tar, rewriting_map):
    src = src.data.cpu().numpy()
    tar = tar.data.cpu().numpy()
    for i, ch in enumerate(src):
        tar_tuple = tuple(tar[i * 3 : i * 3 + 3])
        rewriting_map[ch].add(tar_tuple)

    return rewriting_map

def build_iters(param_iter):

    ftrain = param_iter['ftrain']
    fvalid = param_iter['fvalid']
    bsz = param_iter['bsz']
    device = param_iter['device']

    examples_train = load_examples(ftrain)

    SRC = torchtext.data.Field(sequential=True, use_vocab=True,
                               pad_token=PAD,
                               unk_token=UNK,
                               eos_token=EOS)

    TAR = torchtext.data.Field(sequential=True, use_vocab=True,
                               pad_token=PAD,
                               unk_token=UNK,
                               eos_token=EOS,
                               init_token=SOS)

    train = Dataset(examples_train, fields=[('src', SRC),
                                      ('tar', TAR)])
    SRC.build_vocab(train)
    TAR.build_vocab(train)

    examples_valid = load_examples(fvalid)
    valid = Dataset(examples_valid, fields=[('src', SRC),
                                      ('tar', TAR)])

    train_iter = torchtext.data.Iterator(train, batch_size=bsz,
                                         sort=False, repeat=False,
                                         sort_key=lambda x: len(x.src),
                                         sort_within_batch=True,
                                         device=device)
    valid_iter = torchtext.data.Iterator(valid, batch_size=bsz,
                                         sort=False, repeat=False,
                                         sort_key=lambda x: len(x.src),
                                         sort_within_batch=True,
                                         device=device)

    rewriting_map = defaultdict(set)
    for batch in train_iter:
        src = batch.src
        tar = batch.tar

        mask_src = src.data.eq(SRC.vocab.stoi[PAD])
        mask_tar = tar.data.eq(TAR.vocab.stoi[PAD])
        lens_src = src.shape[0] - mask_src.sum(dim=0)
        lens_tar = tar.shape[0] - mask_tar.sum(dim=0)
        for (l_src, l_tar, b) in zip(lens_src, lens_tar, range(bsz)):
            record_map(src[:l_src-1, b], tar[1:l_tar-1, b], rewriting_map)

    return {'train_iter': train_iter,
            'valid_iter': valid_iter,
            'SRC': SRC,
            'TAR': TAR,
            'rewriting_map': rewriting_map}

def valid_one(src, pred, tar, rewriting_map, src_itos, tar_itos):
    src = src.data.cpu().numpy()
    pred = pred.data.cpu().numpy()
    tar = tar.data.cpu().numpy()

    if pred[-1] != tar[-1]:
        return 0, len(src[:-1])

    nt = 0
    nc = 0
    # print(' '.join([src_itos[i] for i in src]))

    # take out of <eos>
    src = src[:-1]
    pred = pred[:-1]
    tar = tar[:-1]
    for (i, ch) in enumerate(src):
        pred_tuple = tuple(pred[i * 3: i * 3 + 3])
        tar_tuple = tuple(tar[i * 3: i * 3 + 3])

        if tar_tuple not in rewriting_map[ch]:
            print(' '.join([src_itos[i] for i in src]))
            print(' '.join([tar_itos[i] for i in tar]))
            print(ch)
            print(src_itos[ch])
            print(tar_tuple)
            print(tar_itos[tar_tuple[0]])
            print(rewriting_map[ch])

        assert tar_tuple in rewriting_map[ch], 'illegal target'

        if pred_tuple in rewriting_map[ch]:
            nc += 1
        nt += 1

    return nc, nt

def valid(model, valid_iter, rewriting_map):
    nc = 0
    nt = 0
    src_itos = valid_iter.dataset.fields['src'].vocab.itos
    tar_itos = valid_iter.dataset.fields['tar'].vocab.itos
    with torch.no_grad():
        model.eval()
        for i, batch in enumerate(valid_iter):
            src, tar = batch.src, batch.tar

            # take out <sos>
            tar = tar[1:]
            res = model(src, tar)
            outputs = res['outputs']
            pred = outputs.max(dim=-1)[1]

            mask_src = src.data.eq(model.padding_idx_enc)
            mask_tar = tar.data.eq(model.padding_idx_dec)
            lens_src = src.shape[0] - mask_src.sum(dim=0)
            lens_tar = tar.shape[0] - mask_tar.sum(dim=0)
            bsz = src.shape[1]
            for (l_src, l_tar, b) in zip(lens_src, lens_tar, range(bsz)):
                nc_one, nt_one = valid_one(src[:l_src, b],
                                 pred[:l_tar, b],
                                 tar[:l_tar, b],
                                 rewriting_map, src_itos, tar_itos)
                nc += nc_one
                nt += nt_one

    return nc/nt

def train(model, iters, opt, criterion, optim, scheduler):
    train_iter = iters['train_iter']
    valid_iter = iters['valid_iter']
    rewriting_map = iters['rewriting_map']

    basename = "{}-{}-{}-{}-{}".format(opt.task,
                                    opt.sub_task,
                                    opt.enc_type,
                                    opt.dec_type,
                                    utils.time_int())
    log_fname = basename + ".json"
    log_path = os.path.join(RES, log_fname)
    with open(log_path, 'w') as f:
        f.write(str(utils.param_str(opt)) + '\n')
    # print(valid(model, valid_iter, rewriting_map))

    losses = []
    for epoch in range(opt.nepoch):
        for i, batch in enumerate(train_iter):
            src, tar = batch.src, batch.tar
            model.train()
            model.zero_grad()

            # take out <sos>
            tar = tar[1:]
            res = model(src, tar)
            outputs = res['outputs']
            dec_voc_size = model.dec_voc_size
            loss = criterion(outputs.view(-1, dec_voc_size), tar.view(-1))
            losses.append(loss.item())
            loss.backward()
            clip_grad_norm(model.parameters(), 5)
            optim.step()
            loss = {'trans_loss': loss.item()}

            utils.progress_bar(i / len(train_iter), loss, epoch)
            if (i + 1) % int(1 / 4 * len(train_iter)) == 0:
                accurracy = valid(model, valid_iter, rewriting_map)
                loss_ave = np.array(losses).sum() / len(losses)
                losses = []
                log_str = '{\'Epoch\':%d, \'Format\':\'a/l\', \'Metrics\':[%.4f, %.4f]}' % \
                          (epoch, accurracy, loss_ave)

                lr = 0
                for param_group in optim.param_groups:
                    lr = param_group['lr']

                print(log_str, 'lr:', lr)
                with open(log_path, 'a+') as f:
                    f.write(log_str + '\n')

                scheduler.step(accurracy)
                for param_group in optim.param_groups:
                    print('learning rate:', param_group['lr'])

        if (epoch + 1) % opt.save_per == 0:
            model_fname = basename + ".model"
            save_path = os.path.join(RES, model_fname)
            print('Saving to ' + save_path)
            torch.save(model.state_dict(), save_path)

class Model(nn.Module):

    def __init__(self, encoder, decoder,
                 embedding_enc, embedding_dec,
                 sos_idx):

        super(Model, self).__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.embedding_enc = embedding_enc
        self.embedding_dec = embedding_dec
        self.dec_voc_size = self.embedding_dec.num_embeddings
        self.hdim = self.encoder.hdim
        self.padding_idx_enc = embedding_enc.padding_idx
        self.padding_idx_dec = embedding_dec.padding_idx
        self.dec_h0 = nn.Parameter(torch.LongTensor([sos_idx]),
                                   requires_grad=False)
        self.clf = nn.Linear(self.hdim, self.dec_voc_size)

    def forward(self, src, tar):
        mask = src.data.eq(self.padding_idx_enc)
        len_total, bsz = src.shape
        lens = len_total - mask.sum(dim=0)

        embs = self.embedding_enc(src)
        input = {'embs':embs,
                 'lens':lens}
        res = self.encoder(input)

        hid = res['hid']
        enc_outputs = res['output']
        ntm_states = None
        if 'ntm_states' in res.keys():
            ntm_states = res['ntm_states']

        inp = self.dec_h0.expand(1, bsz)
        inp = self.embedding_dec(inp)
        outputs = []
        for target in tar:
            input = {'inp': inp,
                     'hid': hid,
                     'enc_outputs': enc_outputs,
                     'ntm_states': ntm_states}

            res = self.decoder(input)
            output = res['output']
            output = self.clf(output)
            if self.training:
                inp = self.embedding_dec(target.unsqueeze(0))
            else:
                pred_idx = output.max(dim=-1)[1]
                inp = self.embedding_dec(pred_idx)
            hid = res['hid']
            if 'ntm_states' in res.keys():
                ntm_states = res['ntm_states']
            else:
                ntm_states = None
            outputs.append(output)

        outputs = torch.cat(outputs, dim=0)
        return {'outputs':outputs}



