import torchtext
from macros import *
from torchtext.data import Dataset
import torch
from torch import nn
from torch.nn.utils import clip_grad_norm
import numpy as np
import utils
from sklearn.metrics import accuracy_score, \
    precision_score, recall_score, f1_score

class Example(object):

    def __init__(self, src, tar):
        self.src = self.tokenizer(src)
        self.tar = self.tokenizer(tar)

    def tokenizer(self, seq):
        res = seq.split()
        return res

def load_examples(fname):
    examples = []

    with open(fname, 'r') as f:
        for line in f:
            src, tar = line.split(' OUT: ')
            _, src = src.split('IN: ')

            examples.append(Example(src, tar))

    return examples

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

    return {'train_iter': train_iter,
            'valid_iter': valid_iter,
            'SRC': SRC,
            'TAR': TAR}

def valid(model, valid_iter):
    nc = 0
    nt = 0
    with torch.no_grad():
        model.eval()
        for i, batch in enumerate(valid_iter):
            src, tar = batch.src, batch.tar

            # take out <sos>
            tar = tar[1:]
            res = model(src, tar)
            outputs = res['outputs']
            pred = outputs.max(dim=-1)[1]

            max_length, bsz = tar.shape
            mask = tar.data.eq(model.padding_idx_dec)
            lens = max_length - mask.sum(dim=0)
            for (l, b) in zip(lens, range(bsz)):
                cost = torch.abs(pred[:l, b] - tar[:l, b]).sum().item()
                if cost == 0:
                    nc += 1
                nt += 1

    return nc/nt

def train(model, iters, opt, criterion, optim):
    train_iter = iters['train_iter']
    valid_iter = iters['valid_iter']

    basename = "{}-{}-{}-{}-{}".format(opt.task,
                                       opt.sub_task,
                                       opt.enc_type,
                                       opt.dec_type,
                                       utils.time_int())
    log_fname = basename + ".json"
    log_path = os.path.join(RES, log_fname)
    with open(log_path, 'w') as f:
        f.write(str(utils.param_str(opt)) + '\n')
    # print(valid(model, valid_iter))

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
                # print('\r')
                accurracy = valid(model, valid_iter)
                loss_ave = np.array(losses).sum() / len(losses)
                losses = []
                log_str = '{\'Epoch\':%d, \'Format\':\'a/l\', \'Metrics\':[%.4f, %.4f]}' % \
                          (epoch, accurracy, loss_ave)
                print(log_str)
                with open(log_path, 'a+') as f:
                    f.write(log_str + '\n')

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
        self.hdim_dec = self.decoder.hdim
        self.padding_idx_enc = embedding_enc.padding_idx
        self.padding_idx_dec = embedding_dec.padding_idx
        self.dec_inp0 = nn.Parameter(torch.LongTensor([sos_idx]),
                                   requires_grad=False)
        self.clf = nn.Linear(self.hdim_dec, self.dec_voc_size)

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
        nse_states = None
        stack = None
        tardis_states = None
        if 'ntm_states' in res.keys():
            ntm_states = res['ntm_states']
        if 'nse_states' in res.keys():
            nse_states = res['nse_states']
        if 'stack' in res.keys():
            stack = res['stack']
        if 'tardis_states' in res.keys():
            tardis_states = res['tardis_states']

        inp = self.dec_inp0.expand(1, bsz)
        inp = self.embedding_dec(inp)

        outputs = []
        for target in tar:
            input = {'inp': inp,
                     'hid': hid,
                     'enc_outputs': enc_outputs,
                     'ntm_states': ntm_states,
                     'nse_states': nse_states,
                     'stack': stack,
                     'tardis_states': tardis_states}

            res = self.decoder(input)
            output = res['output']
            output = self.clf(output)
            if self.training:
                inp = self.embedding_dec(target.unsqueeze(0))
            else:
                pred_idx = output.max(dim=-1)[1]
                inp = self.embedding_dec(pred_idx)
            hid = res['hid']
            ntm_states = res['ntm_states'] \
                if 'ntm_states' in res.keys() else None
            nse_states = res['nse_states'] \
                if 'nse_states' in res.keys() else None
            stack = res['stack'] \
                if 'stack' in res.keys() else None
            tardis_states = res['tardis_states'] \
                if 'tardis_states' in res.keys() else None

            outputs.append(output)

        outputs = torch.cat(outputs, dim=0)
        return {'outputs':outputs}



