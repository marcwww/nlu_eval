import torch
import utils
from torch.nn.utils import clip_grad_norm
from sklearn.metrics import \
    accuracy_score, precision_score, \
    recall_score, f1_score
import os
from macros import *

def valid_rte(model, valid_iter):
    pred_lst = []
    true_lst = []

    with torch.no_grad():
        model.eval()
        for i, batch in enumerate(valid_iter):
            idx, seq1, seq2, lbl = \
                batch.idx, batch.seq1, batch.seq2, batch.lbl

            # len_total, bsz = seq1.shape
            res = model(seq1, seq2)

            pred = res.max(dim=1)[1].cpu().numpy()
            lbl = lbl.cpu().numpy()
            pred_lst.extend(pred)
            true_lst.extend(lbl)

    accurracy = accuracy_score(true_lst, pred_lst)
    precision = precision_score(true_lst, pred_lst, average='macro')
    recall = recall_score(true_lst, pred_lst, average='macro')
    f1 = f1_score(true_lst, pred_lst, average='macro')

    return accurracy, precision, recall, f1

def train_rte(model, iters, opt, criterion, optim):
    train_iter = iters['train']
    valid_iter = iters['valid']

    print(valid_rte(model, valid_iter))
    for epoch in range(opt.nepoch):
        for i, batch in enumerate(train_iter):
            idx, seq1, seq2, lbl = \
                batch.idx, batch.seq1, batch.seq2, batch.lbl
            model.train()
            model.zero_grad()
            # len_total, bsz = seq1.shape
            res = model(seq1, seq2)

            loss = criterion(res.view(-1, model.nclasses), lbl)
            loss.backward()
            clip_grad_norm(model.parameters(), 5)
            optim.step()

            loss = {'clf_loss':loss.item()}

            utils.progress_bar(i/len(train_iter), loss, epoch)

            if (i + 1) % int(1 / 4 * len(train_iter)) == 0:
                # print('\r')
                accurracy, precision, recall, f1 = \
                    valid_rte(model, valid_iter)
                print('{\'Epoch\':%d, \'Format\':\'a/p/r/f\', \'Metrics\':[%.4f, %.4f, %.4f, %.4f]}' %
                      (epoch, accurracy, precision, recall, f1))

        if (epoch + 1) % opt.save_per == 0:
            basename = "{}-epoch-{}".format(opt.name, epoch)
            model_fname = basename + ".model"
            save_path = os.path.join(RES, model_fname)
            print('Saving to ' + save_path)
            torch.save(model.state_dict(), save_path)