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

    def __init__(self, seq, lbl):
        self.seq = seq
        self.lbl = lbl

def tokenizer(seq):
    return list(seq)

def load_examples(fname):
    examples = []

    with open(fname, 'r') as f:
        for line in f:
            idx, seq1, seq2, lbl = \
                line.split('\t')
            seq = seq1 + SEP + seq2
            examples.append(Example(seq, lbl))

    return examples

def build_iters(ftrain, fvalid, bsz, device):

    examples_train = load_examples(ftrain)

    SEQ = torchtext.data.Field(sequential=True, use_vocab=True,
                               tokenize=tokenizer,
                               pad_token=PAD,
                               unk_token=UNK,
                               eos_token=None)
    LBL = torchtext.data.Field(sequential=False, use_vocab=True)

    train = Dataset(examples_train, fields=[('seq', SEQ),
                                      ('lbl', LBL)])
    SEQ.build_vocab(train)
    LBL.build_vocab(train)

    examples_valid = load_examples(fvalid)
    valid = Dataset(examples_valid, fields=[('seq', SEQ),
                                      ('lbl', LBL)])

    train_iter = torchtext.data.Iterator(train, batch_size=bsz,
                                         sort=False, repeat=False,
                                         sort_key=lambda x: len(x.seq),
                                         sort_within_batch=True,
                                         device=device)
    valid_iter = torchtext.data.Iterator(valid, batch_size=bsz,
                                         sort=False, repeat=False,
                                         sort_key=lambda x: len(x.seq),
                                         sort_within_batch=True,
                                         device=device)

    return {'train_iter': train_iter,
            'valid_iter': valid_iter,
            'SEQ': SEQ,
            'LBL': LBL}

def valid(model, valid_iter):
    pred_lst = []
    true_lst = []

    with torch.no_grad():
        model.eval()
        for i, batch in enumerate(valid_iter):
            seq, lbl = batch.seq, batch.lbl
            res= model(seq)
            res_clf = res['res_clf']

            pred = res_clf.max(dim=1)[1].cpu().numpy()
            lbl = lbl.cpu().numpy()
            pred_lst.extend(pred)
            true_lst.extend(lbl)

    accuracy = accuracy_score(true_lst, pred_lst)
    precision = precision_score(true_lst, pred_lst, average='macro')
    recall = recall_score(true_lst, pred_lst, average='macro')
    f1 = f1_score(true_lst, pred_lst, average='macro')

    return accuracy, precision, recall, f1

def train(model, iters, opt, criterion, optim):
    train_iter = iters['train']
    valid_iter = iters['valid']

    print(valid(model, valid_iter))
    for epoch in range(opt.nepoch):
        for i, batch in enumerate(train_iter):
            seq, lbl = batch.seq, batch.lbl
            model.train()
            model.zero_grad()
            res = model(seq)
            res_clf = res['res_clf']
            loss = criterion(res_clf.view(-1, 3), lbl)
            loss.backward()
            clip_grad_norm(model.parameters(), 5)
            optim.step()

            loss = {'clf_loss': loss.item()}

            utils.progress_bar(i / len(train_iter), loss, epoch)

            if (i + 1) % int(1 / 4 * len(train_iter)) == 0:
                # print('\r')
                accurracy, precision, recall, f1 = \
                    valid(model, valid_iter)
                print('{\'Epoch\':%d, \'Format\':\'a/p/r/f\', \'Metrics\':[%.4f, %.4f, %.4f, %.4f]}' %
                      (epoch, accurracy, precision, recall, f1))

        if (epoch + 1) % opt.save_per == 0:
            basename = "{}-epoch-{}".format(opt.task, epoch)
            model_fname = basename + ".model"
            save_path = os.path.join(RES, model_fname)
            print('Saving to ' + save_path)
            torch.save(model.state_dict(), save_path)

class Model(nn.Module):

    def __init__(self, encoder, embedding):
        super(Model, self).__init__()
        self.encoder = encoder
        self.embedding = embedding
        self.hdim = self.encoder.hdim
        self.clf = nn.Linear(self.hdim, 3)
        self.padding_idx = embedding.padding_idx

    def forward(self, input):
        embs = self.embedding(input)
        res = self.encoder(embs)
        mask = input.data.eq(self.padding_idx)
        len_total, bsz = input.shape
        lens = len_total - mask.sum(dim=0)
        output = res['output']
        reps = torch.cat([output[lens[b] - 1, b, :].unsqueeze(0) for b in range(bsz)],
                         dim=0)
        res_clf = self.clf(reps)
        return {'res_clf':res_clf}

criterion = nn.CrossEntropyLoss()



