import torchtext
from macros import *
from torchtext.data import Dataset
import torch
from torch import nn
from torch.nn.utils import clip_grad_norm
import numpy as np
import utils
import random
from sklearn.metrics import accuracy_score, \
    precision_score, recall_score, f1_score

def gen_batch_copy(batch_info):

    min_len = batch_info['min_len']
    max_len = batch_info['max_len']
    bsz = batch_info['bsz']
    seq_width = batch_info['seq_width']

    seq_len = random.randint(min_len, max_len)
    seq = np.random.binomial(1, 0.5, (seq_len, bsz, seq_width))
    seq = torch.Tensor(seq)

    # The input includes an additional channel used for the delimiter
    inp = torch.zeros(seq_len + 1, bsz, seq_width + 1)
    inp[:seq_len, :, :seq_width] = seq
    inp[seq_len, :, seq_width] = 1.0  # delimiter in our control channel
    outp = seq.clone()

    return inp.float(), outp.float()

class Iter(object):

    def __init__(self, gen_batch, batch_info,
                 bsz, device):
        self.gen_batch = gen_batch
        self.batch_info = batch_info
        self.bsz = bsz
        self.num_batches = batch_info['num_batches']
        self.batch_idx = 0
        location = device if torch.cuda.is_available() and \
                             device != -1 else 'cpu'
        self.device = torch.device(location)

    def __iter__(self):
        return self

    def __len__(self):
        return self.num_batches

    def _restart(self):
        self.batch_idx = 0

    def __next__(self):
        if self.batch_idx >= self.num_batches:
            self._restart()
            raise StopIteration()

        inp, outp = self.gen_batch(self.batch_info)
        self.batch_idx += 1
        return inp.to(self.device), \
               outp.to(self.device)

def build_iters(param_iter):

    sub_task = param_iter['sub_task']
    bsz = param_iter['bsz']
    device = param_iter['device']
    num_batches_train = param_iter['num_batches_train']
    min_len_train = param_iter['min_len_train']
    max_len_train = param_iter['max_len_train']
    num_batches_valid = param_iter['num_batches_valid']
    min_len_valid = param_iter['min_len_valid']
    max_len_valid = param_iter['max_len_valid']
    seq_width = param_iter['seq_width']

    train_iter = None
    valid_iter = None
    if sub_task == 'copy':
        train_info = {'num_batches':num_batches_train,
                      'min_len': min_len_train,
                      'max_len': max_len_train,
                      'bsz': bsz,
                      'seq_width': seq_width}

        valid_info = {'num_batches':num_batches_valid,
                      'min_len': min_len_valid,
                      'max_len': max_len_valid,
                      'bsz': bsz,
                      'seq_width': seq_width}

        train_iter = Iter(gen_batch_copy, train_info, bsz, device)
        valid_iter = Iter(gen_batch_copy, valid_info, bsz, device)

    return {'train_iter': train_iter,
            'valid_iter': valid_iter}

def valid(model, valid_iter):
    nc = 0
    nt = 0
    with torch.no_grad():
        model.eval()

        for inp, out_tar in valid_iter:
            out_len = out_tar.shape[0]
            out = model(inp, out_len)
            out = out['res_clf']
            out = out.data.gt(0.5).float()
            bsz = out.shape[1]

            for b in range(bsz):
                cost = torch.sum(torch.abs(out[:,b,:] - out_tar[:,b,:]))
                if cost == 0:
                    nc += 1
                nt += 1

    return nc/nt

def train(model, iters, opt, criterion, optim):
    train_iter = iters['train_iter']
    valid_iter = iters['valid_iter']

    # print(valid(model, valid_iter))
    for epoch in range(opt.nepoch):
        for i, (inp, out_tar) in enumerate(train_iter):
            model.train()
            model.zero_grad()
            out_len = out_tar.shape[0]
            out = model(inp, out_len)
            out = out['res_clf']

            loss = criterion(out, out_tar)
            loss.backward()
            clip_grad_norm(model.parameters(), 5)
            optim.step()

            loss = {'clf_loss': loss.item()}

            utils.progress_bar(i / len(train_iter), loss, epoch)

            if (i + 1) % int(1 / 4 * len(train_iter)) == 0:
                # print('\r')
                accurracy = \
                    valid(model, valid_iter)
                print('{\'Epoch\':%d, \'Format\':\'a\', \'Metrics\':[%.4f]}' %
                      (epoch, accurracy))

        if (epoch + 1) % opt.save_per == 0:
            basename = "{}-epoch-{}".format(opt.task, epoch)
            model_fname = basename + ".model"
            save_path = os.path.join(RES, model_fname)
            print('Saving to ' + save_path)
            torch.save(model.state_dict(), save_path)

class Model(nn.Module):

    def __init__(self, encoder, odim):
        super(Model, self).__init__()
        self.encoder = encoder
        self.hdim = self.encoder.hdim
        self.odim = odim
        self.clf = nn.Sequential(nn.Linear(self.hdim, self.odim),
                                 nn.Sigmoid())

    def forward(self, seq, out_len):
        in_len, bsz, idim = seq.shape
        device = seq.device
        pad = torch.zeros(out_len, bsz, idim).to(device)
        inp = torch.cat([seq, pad], dim=0)
        lens = torch.LongTensor([inp.shape[0]] * bsz).to(device)
        input = {'embs': inp, 'lens':lens}

        out = self.encoder(input)

        out = out['output']
        out = out[in_len:]
        out = self.clf(out)

        return {'res_clf':out}

criterion = nn.BCELoss()



