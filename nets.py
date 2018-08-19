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
        # self.gru = nn.GRU(hidden_size, hidden_size)
        self.hid2hid = nn.Linear(hdim, hdim)
        self.emb2hid = nn.Linear(edim, hdim)
        self.hid2act = nn.Linear(hdim, len(ACTS))
        self.hid2gamma = nn.Linear(hdim, 1)
        self.hid2stack = nn.Linear(hdim, sdim)
        self.stack2hid = nn.Linear(sdim * self.sdepth, hdim)
        self.stack2u = nn.Linear(sdim * self.sdepth, sdim)

        self.empty_elem = nn.Parameter(torch.Tensor(1, self.sdim))

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

    def update_stack(self, stack,
                     p_push, p_pop, p_noop,
                     push_val, u_val):

        # stack: (bsz, ssz, sdim)
        # p_push, p_pop, p_noop: (bsz, 1)
        # push_val: (bsz, sdim)
        # p_xact: bsz * nstack
        p_push = p_push.unsqueeze(-1)
        p_pop = p_pop.unsqueeze(-1)
        p_noop = p_noop.unsqueeze(-1)

        bsz, ssz, sdim  = stack.shape
        stack_push = self.W_push.matmul(stack)
        stack_push[:, 0, :] += push_val

        stack_pop = self.W_pop.matmul(stack)
        stack_pop = self.W_push.matmul(stack_pop)
        stack_pop[:, 0, :] += u_val

        stack_noop = stack

        stack  = p_push * stack_push + p_pop * stack_pop + p_noop * stack_noop
        return stack

    def forward(self, inputs):

        bsz = inputs.shape[1]
        hid = self.zero.expand(bsz, self.hdim)
        stack = self.empty_elem.expand(bsz,
                                      self.ssz,
                                      self.sdim)

        # inputs: (length, bsz)
        # stacks: (bsz, ssz, sdim)
        # embs: (length, bsz, edim)
        embs = self.embedding(inputs)
        # inputs(length,bsz)->embd(length,bsz,embdsz)

        outputs = []
        acts = []
        for emb in embs:
            # stack_vals: (bsz, stack_depth * sdim)
            # catenate all the readed vectors:
            tops = stack[:, :self.sdepth, :].contiguous(). \
                view(bsz,
                     self.sdepth * self.sdim)

            # emb: (bsz, embdsz)
            mhid= self.emb2hid(emb) + self.hid2hid(hid) + self.stack2hid(tops)

            # act: (bsz, nacts)
            act = self.hid2act(hid)
            gamma = self.hid2gamma(hid)
            gamma = 1+torch.log(1+torch.exp(gamma))
            act = F.softmax(act, dim=-1)
            act_sharpened = act ** gamma
            act_sharpened= torch.div(act_sharpened, torch.sum(act_sharpened, dim=-1).view(-1, 1) + 1e-16)

            # act = F.gumbel_softmax(act, tau=self.tau)

            # p_push, p_pop, p_noop: (bsz, 1)
            p_push, p_pop, p_noop = act_sharpened.chunk(len(ACTS), dim=-1)
            _, act_chosen = torch.topk(act_sharpened, k=1, dim=-1)
            acts.append(act_chosen.unsqueeze(0))

            # push_vals: (bsz, sdim)
            push_val = self.hid2stack(hid)

            # push_val: (bsz, ssz)
            push_val = self.nonLinear(push_val)
            # u_val: (bsz, ssz) unified stack element
            u_val = self.nonLinear(self.stack2u(tops))
            stack = self.update_stack(stack,
                                       p_push, p_pop, p_noop,
                                       push_val, u_val)

            hid = self.nonLinear(mhid)
            outputs.append(hid.unsqueeze(0))

        outputs = torch.cat(outputs, dim=0)
        acts = torch.cat(acts, dim=0).squeeze(-1)

        return {'outputs':outputs,
                'hid':hid,
                'stack':stack,
                'act':acts}

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
        lens1 = len_total1 - mask1.sum(dim=0)
        mask2 = seq2.data.eq(self.padding_idx)
        lens2 = len_total2 - mask2.sum(dim=0)

        # output: (len_total, bsz, hdim)
        lens1 = torch.LongTensor(lens1.data.cpu())
        lens2 = torch.LongTensor(lens2.data.cpu())
        outputs1 = res1['outputs']
        outputs2 = res2['outputs']
        fhid1 = torch.cat([outputs1[l - 1, b, :].unsqueeze(0) for l, b in zip(lens1, range(bsz1))],
                          dim=0)
        fhid2 = torch.cat([outputs2[l - 1, b, :].unsqueeze(0) for l, b in zip(lens2, range(bsz2))],
                          dim=0)

        u1 = fhid1
        u2 = fhid2
        u3 = torch.abs(fhid1 - fhid2)
        u4 = fhid1 * fhid2

        u = torch.cat([u1, u2, u3, u4], dim=-1)
        res = self.clf(u)

        return res









