import torch
import utils
from torch.nn.utils import clip_grad_norm
from sklearn.metrics import \
    accuracy_score, precision_score, \
    recall_score, f1_score
import os
import numpy as np
from macros import *

def explain(txt, acts):
    # acts: (seq_len, nacts)
    example = []
    # - 1 for <eos>
    N = len(txt) - 1
    T = 2 * N - 1
    for t in range(T):
        if t >= len(txt):
            break
        # w = txt[t] if t < len(txt) else PAD
        w = txt[t]
        pact = acts[t]
        w_str = '%s (%.2f, %.2f)' % \
                (w, pact[0].item(), pact[1].item())
        example.append(w_str)
        _, chosen_act = torch.topk(acts[t], k=1)
        if chosen_act == ACTS['pop']:
            txt.insert(t + 1, w)

    return '|'.join(example)

def valid_rte(model, valid_iter):
    pred_lst = []
    true_lst = []
    txt1_lst = []
    txt2_lst = []
    acts1_lst = []
    acts2_lst = []
    lbl_lst = []

    with torch.no_grad():
        model.eval()
        for i, batch in enumerate(valid_iter):
            indices, seq1, seq2, lbl = \
                batch.idx, batch.seq1, batch.seq2, batch.lbl

            bsz = indices.shape[0]

            for i in range(bsz):
                example = valid_iter.dataset.examples[indices[i].item()]
                idx, txt1, txt2, lbl_str = example.idx, \
                                           example.seq1, \
                                           example.seq2, \
                                           example.lbl

                assert indices[i].item() == int(idx), 'indices not match'
                txt1_lst.append(txt1)
                txt2_lst.append(txt2)
                lbl_lst.append(lbl_str)

            # len_total, bsz = seq1.shape
            res= model(seq1, seq2)
            res_clf = res['res_clf']
            res1 = res['res1']
            res2 = res['res2']

            for i in range(bsz):
                acts1 = res1['act'][:, i, :]
                acts2 = res2['act'][:, i, :]
                acts1_lst.append(acts1)
                acts2_lst.append(acts2)

            pred = res_clf.max(dim=1)[1].cpu().numpy()
            lbl = lbl.cpu().numpy()
            pred_lst.extend(pred)
            true_lst.extend(lbl)

    print('-'*20 + '4 examples begin' + '-'*20)
    indices_chosen = np.random.choice(len(lbl_lst), 4)
    for idx in indices_chosen:
        premise = explain(txt1_lst[idx], acts1_lst[idx])
        hypothesis = explain(txt2_lst[idx], acts2_lst[idx])
        lbl = lbl_lst[idx]
        print('[%d] lbl:%s\n push/pop\n p:%s\n h:%s' % (idx, lbl, premise, hypothesis))
    print('-' * 20 + '4 examples end' + '-' * 20)

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
            res_clf = res['res_clf']

            loss = criterion(res_clf.view(-1, model.nclasses), lbl)

            # dis1 = res['dis1']
            # dis2 = res['dis2']
            # diff1 = res['diff1']
            # diff2 = res['diff2']
            # loss += opt.coef_dis * (dis1 + dis2) + opt.coef_diff * (diff1 + diff2)
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