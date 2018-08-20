import numpy as np
from torch import nn
import torch
from macros import *
from torch.nn import functional as F
import utils

class EncoderSRNN(nn.Module):
    def __init__(self, voc_size, edim, hdim,
                 stack_size, sdim, padding_idx,
                 fine_tuning, stack_depth = 2):
        super(EncoderSRNN, self).__init__()
        # here input dimention is equal to hidden dimention
        self.edim = edim
        self.hdim = hdim
        self.ssz = stack_size
        self.sdepth = stack_depth
        self.sdim = sdim
        self.padding_idx = padding_idx
        self.embedding = nn.Embedding(voc_size,
                                      edim,
                                      padding_idx=padding_idx)
        self.embedding.weight.requires_grad = fine_tuning
        self.nonLinear=nn.ReLU()

        self.hid2hid = nn.Linear(hdim, hdim)
        self.emb2hid = nn.Linear(edim, hdim)

        self.hid2inst = nn.Linear(hdim, len(ACTS) + 1)

        self.hid2stack = nn.Linear(hdim, sdim)
        self.stack2hid = nn.Linear(sdim * self.sdepth, hdim)
        self.stack2u = nn.Linear(sdim * self.sdepth, sdim)

        self.empty_elem = nn.Parameter(torch.Tensor(1, self.sdim))

        # shift matrix for stack
        W_up, W_down = utils.shift_matrix(stack_size)
        W_pop = W_up
        for i in range(self.sdepth - 1):
            W_pop = np.matmul(W_pop, W_up)
        self.W_up = nn.Parameter(torch.Tensor(W_up), requires_grad=False)
        self.W_pop = nn.Parameter(torch.Tensor(W_pop), requires_grad=False)

        self.W_down = nn.Parameter(torch.Tensor(W_down), requires_grad=False)
        self.W_push = self.W_down

        self.tau = nn.Parameter(torch.Tensor(1).uniform_(0, 1))
        # self.tau = 1
        self.zero = nn.Parameter(torch.zeros(1), requires_grad=False)
        self.pad = nn.Parameter(torch.LongTensor([padding_idx]), requires_grad=False)

    def update_stack(self, stack,
                     p_push, p_pop,
                     push_val, u_val):

        # stack: (bsz, ssz, sdim)
        # p_push, p_pop, p_noop: (bsz, 1)
        # push_val: (bsz, sdim)
        # p_xact: bsz * nstack
        p_push = p_push.unsqueeze(-1)
        p_pop = p_pop.unsqueeze(-1)

        bsz, ssz, sdim  = stack.shape
        stack_push = self.W_push.matmul(stack)
        stack_push[:, 0, :] += push_val

        stack_pop = self.W_pop.matmul(stack)
        stack_pop = self.W_push.matmul(stack_pop)
        stack_pop[:, 0, :] += u_val
        # fill the stack with empty elements
        stack_pop[:, self.ssz - self.sdepth + 1:, :] += self.empty_elem

        stack  = p_push * stack_push + p_pop * stack_pop
        return stack

    def update_buf(self, buf,
                   p_push, p_pop):

        device = buf.device
        T = buf.shape[1]
        W_up, W_down = utils.shift_matrix(T)
        W_down = torch.Tensor(W_down).to(device)

        # buf: (bsz, T, edim)
        p_push = p_push.unsqueeze(-1)
        p_pop = p_pop.unsqueeze(-1)

        buf_push = buf
        buf_pop = W_down.matmul(buf)
        # buf_noop = buf

        buf = p_push * buf_push + p_pop * buf_pop
        return buf

    def forward(self, inputs):

        bsz = inputs.shape[1]
        hid = self.zero.expand(bsz, self.hdim)
        stack = self.empty_elem.expand(bsz,
                                      self.ssz,
                                      self.sdim)

        # inputs: (N, bsz)
        # stacks: (bsz, ssz, sdim)
        # embs: (length, bsz, edim)
        N = len(inputs)
        T = 2 * N -1
        pads = self.pad.expand(T - N, bsz)
        pads = self.embedding(pads)
        embs = self.embedding(inputs)

        # bufs: (T, bsz, edim)
        buf = torch.cat([embs, pads], dim=0)

        outputs = []
        top_elems = []
        acts = []
        for t in range(T):
            # stack_vals: (bsz, stack_depth * sdim)
            # catenate all the readed vectors:
            tops = stack[:, :self.sdepth, :].contiguous(). \
                view(bsz,
                     self.sdepth * self.sdim)

            input = buf[t]
            # emb: (bsz, embdsz)
            mhid= self.emb2hid(input) + self.hid2hid(hid) + self.stack2hid(tops)

            # act: (bsz, nacts)
            # act = self.hid2act(hid)
            # inst: (bsz, nacts + 1) probability of actions and gamma
            inst = self.hid2inst(hid)
            act = inst[:, :len(ACTS)]
            gamma = inst[:, len(ACTS):]

            gamma = 1 + torch.log(1 + torch.exp(gamma))
            act = F.softmax(act, dim=-1)
            act_sharpened = act ** gamma
            act_sharpened = torch.div(act_sharpened, torch.sum(act_sharpened, dim=-1).view(-1, 1) + 1e-16)

            # act = F.gumbel_softmax(act, tau=self.tau)

            # p_push, p_pop, p_noop: (bsz, 1)
            p_push, p_pop = \
                act_sharpened.chunk(len(ACTS), dim=-1)

            buf = self.update_buf(buf.transpose(0, 1), p_push, p_pop).\
                transpose(0, 1)

            # _, act_chosen = torch.topk(act_sharpened, k=1, dim=-1)
            acts.append(act.unsqueeze(0))

            # push_val: (bsz, sdim)
            push_val = self.hid2stack(hid)

            # push_val: (bsz, ssz)
            push_val = self.nonLinear(push_val)
            # u_val: (bsz, ssz) unified stack element
            u_val = self.nonLinear(self.stack2u(tops))
            stack = self.update_stack(stack,
                                       p_push, p_pop,
                                       push_val, u_val)
            top_elem = stack[:, 0, :].unsqueeze(0)
            top_elems.append(top_elem)

            hid = self.nonLinear(mhid)
            outputs.append(hid.unsqueeze(0))

        outputs = torch.cat(outputs, dim=0)
        # acts: (T, bsz, nacts)
        acts = torch.cat(acts, dim=0)
        # top_elems: (T, bsz, sdim)
        top_elems = torch.cat(top_elems, dim=0)

        return {'outputs':outputs,
                'hid':hid,
                'stack':stack,
                'act':acts,
                'top':top_elems}

class TextualEntailmentModel(nn.Module):

    def __init__(self, encoder, nclasses):
        super(TextualEntailmentModel, self).__init__()
        self.encoder = encoder
        self.padding_idx = encoder.padding_idx
        self.clf = nn.Sequential(nn.Linear(4 * encoder.hdim, encoder.hdim),
                                 nn.ReLU(),
                                 nn.Linear(encoder.hdim, nclasses))
        self.nclasses = nclasses

    def forward(self, seq1, seq2):
        res1 = self.encoder(seq1)
        res2 = self.encoder(seq2)
        len_total1, bsz1 = seq1.shape
        len_total2, bsz2 = seq2.shape
        mask1 = seq1.data.eq(self.padding_idx)
        # - 1 for <eos>
        lens1 = len_total1 - mask1.sum(dim=0) - 1
        mask2 = seq2.data.eq(self.padding_idx)
        lens2 = len_total2 - mask2.sum(dim=0) - 1

        # output: (len_total, bsz, hdim)
        lens1 = torch.LongTensor(lens1.data.cpu())
        lens2 = torch.LongTensor(lens2.data.cpu())
        Ts1 = 2 * lens1 - 1
        Ts2 = 2 * lens2 - 1

        top_elems1 = res1['top']
        top_elems2 = res2['top']
        fhid1 = torch.cat([top_elems1[T - 1, b, :].unsqueeze(0) for T, b in zip(Ts1, range(bsz1))],
                          dim=0)
        fhid2 = torch.cat([top_elems2[T - 1, b, :].unsqueeze(0) for T, b in zip(Ts2, range(bsz2))],
                          dim=0)

        u1 = fhid1
        u2 = fhid2
        u3 = torch.abs(fhid1 - fhid2)
        u4 = fhid1 * fhid2

        u = torch.cat([u1, u2, u3, u4], dim=-1)
        res_clf = self.clf(u)


        # acts: (T, bsz, nacts)
        acts1 = res1['act']
        # sum_push: (bsz, 1)
        sum_push1 = torch.cat([(acts1[:T, b, ACTS['push']].sum(dim=0) / T.item()).unsqueeze(0) for T, b in zip(Ts1, range(bsz1))],
                             dim=0)
        sum_pop1 = torch.cat([((acts1[:T, b, ACTS['pop']].sum(dim=0) + 1) / T.item()).unsqueeze(0) for T, b in zip(Ts1, range(bsz1))],
                             dim=0)
        # diff: (bsz, 1)
        diff1 = torch.cat(
            [(torch.norm((acts1[:T, b, ACTS['push']] - acts1[:T, b, ACTS['pop']]), dim=0) / T.item()).unsqueeze(0)
                for T, b in zip(Ts1, range(bsz1))], dim=0)

        acts2 = res2['act']
        sum_push2 = torch.cat([(acts2[:T, b, ACTS['push']].sum(dim=0) / T.item()).unsqueeze(0) for T, b in zip(Ts2, range(bsz2))],
                             dim=0)
        sum_pop2 = torch.cat([((acts2[:T, b, ACTS['pop']].sum(dim=0).unsqueeze(0) + 1) / T.item()) for T, b in zip(Ts2, range(bsz2))],
                            dim=0)
        diff2 = torch.cat(
            [(torch.norm((acts2[:T, b, ACTS['push']] - acts2[:T, b, ACTS['pop']]), dim=0) / T.item()).unsqueeze(0)
                for T, b in zip(Ts2, range(bsz2))], dim=0)

        dis1 = torch.norm(sum_push1 - sum_pop1, p=2)/bsz1
        dis2 = torch.norm(sum_push2 - sum_pop2, p=2)/bsz2

        diff1 = diff1.sum() / bsz1
        diff2 = diff2.sum() / bsz2

        return res1, res2, res_clf, dis1, dis2, diff1, diff2









