import torchtext
from macros import *
from torchtext.data import Dataset
import torch
from torch import nn
from torch.nn.utils import clip_grad_norm
import numpy as np
import utils
import itertools
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

def build_iters(ftrain, fvalid, bsz, device):

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

    src_itos = SRC.vocab.itos
    src_stoi = SRC.vocab.stoi
    tar_itos = TAR.vocab.itos
    tar_stoi = TAR.vocab.stoi

    rewriting_map = {}
    for i, ch in enumerate(src_itos):
        if ch in [PAD, UNK, EOS, SOS]:
            continue
        # if i == 39:
        #     print('aaa')

        num = alpha_order(ch[0])
        if num > 19:
            num -= 1
        if len(ch) > 1:
            num += 25

        # if num == 25:
        #     print('aaa')

        candis = set()
        for combi in range(8):
            a = int(combi / 4) % 2
            b = int(combi / 2) % 2
            c = combi % 2

            A = 'A%d_%d' % (num, a + 1)
            B = 'B%d_%d' % (num, b + 1)
            C = 'C%d_%d' % (num, c + 1)

            perms = itertools.permutations([tar_stoi[A], tar_stoi[B], tar_stoi[C]], 3)
            for candi in perms:
                candis.add(candi)
        rewriting_map[i] = candis

    return {'train_iter': train_iter,
            'valid_iter': valid_iter,
            'SRC': SRC,
            'TAR': TAR,
            'rewriting_map': rewriting_map}

def valid_one(src, pred, tar, rewriting_map, src_itos, tar_itos):
    src = src.data.cpu().numpy()
    pred = pred.data.cpu().numpy()
    tar = tar.data.cpu().numpy()

    # if pred[-1] != tar[-1]:
    #     return 0

    src = src[:-1]
    pred = pred[1:-1]
    tar = tar[1:-1]
    for (i, ch) in enumerate(src):
        pred_tuple = tuple(pred[i * 3: i * 3 + 3])
        tar_tuple = tuple(tar[i * 3: i * 3 + 3])

        if tar_tuple not in rewriting_map[ch]:
            print(ch)
            print(src_itos[ch])
            print(tar_tuple)
            print(tar_itos[tar_tuple[0]])
            print(rewriting_map[ch])

        assert tar_tuple in rewriting_map[ch], 'illegal target'

        if pred_tuple not in rewriting_map[ch]:
            return 0

    return 1

def valid(model, valid_iter, rewriting_map):
    nc = 0
    nt = 0
    src_itos = valid_iter.dataset.fields['src'].vocab.itos
    tar_itos = valid_iter.dataset.fields['tar'].vocab.itos
    with torch.no_grad():
        model.eval()
        for i, batch in enumerate(valid_iter):
            src, tar = batch.src, batch.tar
            max_length, bsz = tar.shape
            res = model(src, tar)
            outputs = res['outputs']
            pred = outputs.max(dim=-1)[1]

            mask = tar.data.eq(model.padding_idx)
            lens = max_length - mask.sum(dim=0)
            for (l, b) in zip(lens, range(bsz)):
                valid_res = valid_one(src[:, b],
                                 pred[:l, b],
                                 tar[:l, b],
                                 rewriting_map, src_itos, tar_itos)
                nc += valid_res
                nt += 1

    return nc/nt

def train(model, iters, opt, criterion, optim):
    train_iter = iters['train_iter']
    valid_iter = iters['valid_iter']
    rewriting_map = iters['rewriting_map']

    print(valid(model, valid_iter, rewriting_map))
    for epoch in range(opt.nepoch):
        for i, batch in enumerate(train_iter):
            src, tar = batch.src, batch.tar
            model.train()
            model.zero_grad()
            res = model(src, tar)
            outputs = res['outputs']
            dec_voc_size = model.dec_voc_size
            loss = criterion(outputs.view(-1, dec_voc_size), tar.view(-1))
            loss.backward()
            clip_grad_norm(model.parameters(), 5)
            optim.step()

            loss = {'trans_loss': loss.item()}

            utils.progress_bar(i / len(train_iter), loss, epoch)

            if (i + 1) % int(1 / 4 * len(train_iter)) == 0:
                # print('\r')
                accurracy = valid(model, valid_iter, rewriting_map)
                print('{\'Epoch\':%d, \'Format\':\'a\', \'Metric\':[%.4f]}' %
                      (epoch, accurracy) )

        if (epoch + 1) % opt.save_per == 0:
            basename = "{}-epoch-{}".format(opt.task, epoch)
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
        self.padding_idx = embedding_enc.padding_idx
        self.dec_h0 = nn.Parameter(torch.LongTensor([sos_idx]),
                                   requires_grad=False)
        self.clf = nn.Linear(self.hdim, self.dec_voc_size)

    def forward(self, src, tar):
        mask = src.data.eq(self.padding_idx)
        len_total, bsz = src.shape
        lens = len_total - mask.sum(dim=0)

        embs = self.embedding_enc(src)
        input = {'embs':embs,
                 'lens':lens}
        res = self.encoder(input)

        hid = res['hid']
        inp = self.dec_h0.expand(1, bsz)
        inp = self.embedding_dec(inp)
        outputs = []
        for target in tar:
            res = self.decoder(inp, hid)
            output = res['output']
            output = self.clf(output)
            if self.training:
                inp = self.embedding_dec(target.unsqueeze(0))
            else:
                pred_idx = output.max(dim=-1)[1]
                inp = self.embedding_dec(pred_idx)
            hid = res['hid']
            outputs.append(output)

        outputs = torch.cat(outputs, dim=0)
        return {'outputs':outputs}

criterion = nn.CrossEntropyLoss()



