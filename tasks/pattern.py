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

def build_iters(ftrain, fvalid, bsz, device):

    train_iter = torch.load(ftrain)
    valid_iter = torch.load(fvalid)

    if device != -1:
        for i, inp, outp in train_iter:
            train_iter[i] = (i, inp.to(device), outp.to(device))

        for i, inp, outp in valid_iter:
            valid_iter[i] = (i, inp.to(device), outp.to(device))

    return {'train_iter': train_iter,
            'valid_iter': valid_iter}

def valid(model, valid_iter):
    nc = 0
    nt = 0
    with torch.no_grad():
        model.eval()

        for i, inp, out_tar in valid_iter:
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

    print(valid(model, valid_iter))
    for epoch in range(opt.nepoch):
        for i, inp, out_tar in train_iter:
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



