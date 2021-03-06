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

    def __init__(self, seq1, seq2, lbl):
        self.seq1 = self.tokenizer(seq1)
        self.seq2 = self.tokenizer(seq2)
        self.lbl = int(lbl)

    def tokenizer(self, seq):
        return list(seq)

def load_examples(fname):
    examples = []

    with open(fname, 'r') as f:
        for line in f:
            seq1, seq2, lbl, _, _, _ = \
                line.split(',')
            examples.append(Example(seq1, seq2, lbl))

    return examples

def build_iters(param_iter):

    ftrain = param_iter['ftrain']
    fvalid = param_iter['fvalid']
    bsz = param_iter['bsz']
    device = param_iter['device']

    examples_train = load_examples(ftrain)

    SEQ = torchtext.data.Field(sequential=True, use_vocab=True,
                               pad_token=PAD,
                               unk_token=UNK,
                               eos_token=None)
    LBL = torchtext.data.Field(sequential=False, use_vocab=False)

    train = Dataset(examples_train, fields=[('seq1', SEQ),
                                            ('seq2', SEQ),
                                            ('lbl', LBL)])
    SEQ.build_vocab(train)
    examples_valid = load_examples(fvalid)
    valid = Dataset(examples_valid, fields=[('seq1', SEQ),
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
            seq1, seq2, lbl = batch.seq1, batch.seq2, batch.lbl
            res= model(seq1, seq2)
            res_clf = res['res_clf']

            pred = res_clf.max(dim=1)[1].cpu().numpy()
            lbl = lbl.cpu().numpy()
            pred_lst.extend(pred)
            true_lst.extend(lbl)

    accuracy = accuracy_score(true_lst, pred_lst)
    precision = precision_score(true_lst, pred_lst)
    recall = recall_score(true_lst, pred_lst)
    f1 = f1_score(true_lst, pred_lst)

    return accuracy, precision, recall, f1

def train(model, iters, opt, criterion, optim):
    train_iter = iters['train_iter']
    valid_iter = iters['valid_iter']

    basename = "{}-{}-{}-{}".format(opt.task,
                                       opt.sub_task,
                                       opt.enc_type,
                                       utils.time_int())
    log_fname = basename + ".json"
    log_path = os.path.join(RES, log_fname)
    with open(log_path, 'w') as f:
        f.write(str(utils.param_str(opt)) + '\n')
    # print(valid(model, valid_iter))

    losses = []
    for epoch in range(opt.nepoch):
        for i, batch in enumerate(train_iter):
            seq1, seq2, lbl = batch.seq1, batch.seq2, batch.lbl

            model.train()
            model.zero_grad()
            res = model(seq1, seq2)
            res_clf = res['res_clf']
            loss = criterion(res_clf.view(-1, 2), lbl)
            losses.append(loss.item())
            loss.backward()
            clip_grad_norm(model.parameters(), 5)
            optim.step()

            loss = {'clf_loss': loss.item()}

            utils.progress_bar(i / len(train_iter), loss, epoch)

            if (i + 1) % int(1 / 4 * len(train_iter)) == 0:
                # print('\r')
                accurracy, precision, recall, f1 = \
                    valid(model, valid_iter)
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

    def __init__(self, encoder, embedding):
        super(Model, self).__init__()
        self.encoder = encoder
        self.embedding = embedding
        self.hdim = self.encoder.hdim
        self.clf = nn.Linear(self.hdim * 4, 2)
        self.padding_idx = embedding.padding_idx

    def enc(self, seq,
            ntm_states=None,
            nse_states=None,
            stack=None,
            tardis_states=None,
            enc_outputs=None):

        mask = seq.data.eq(self.padding_idx)
        len_total, bsz = seq.shape
        lens = len_total - mask.sum(dim=0)

        embs = self.embedding(seq)
        input = {'embs': embs,
                 'lens': lens,
                 'ntm_states': ntm_states,
                 'nse_states': nse_states,
                 'stack': stack,
                 'tardis_states': tardis_states,
                 'enc_outputs': enc_outputs}
        res = self.encoder(input)
        output = res['output']
        reps = torch.cat([output[lens[b] - 1, b, :].unsqueeze(0) for b in range(bsz)],
                         dim=0)
        res['reps'] = reps
        return res

    def forward(self, seq1, seq2):
        res1 = self.enc(seq1)
        reps1 = res1['reps']
        ntm_states = None
        nse_states = None
        stack = None
        tardis_states = None
        enc_outputs = None
        if 'ntm_states' in res1.keys() and \
            res1['ntm_states'] is not None:
            ntm_states = res1['ntm_states']
        if 'nse_states' in res1.keys() and \
            res1['nse_states'] is not None:
            nse_states = res1['nse_states']
        if 'stack' in res1.keys() and \
            res1['stack'] is not None:
            stack = res1['stack']
        if 'tardis_states' in res1.keys() and \
            res1['tardis_states'] is not None:
            tardis_states = res1['tardis_states']
        if 'enc_outputs' in res1.keys() and \
            res1['enc_outputs'] is not None:
            enc_outputs = res1['enc_outputs']

        res2 = self.enc(seq2,
                        ntm_states,
                        nse_states,
                        stack,
                        tardis_states,
                        enc_outputs)
        reps2 = res2['reps']

        if type(reps1) == tuple and \
                type(reps2) == tuple:
            reps1 = reps1[0]
            reps2 = reps2[0]

        u1 = reps1
        u2 = reps2
        u3 = reps1 * reps2
        u4 = torch.abs(reps1 - reps2)

        u = torch.cat([u1, u2, u3, u4], dim=-1)
        res_clf = self.clf(u)
        return {'res_clf':res_clf}



