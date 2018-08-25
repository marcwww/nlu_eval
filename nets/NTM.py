import numpy as np
from torch import nn
import torch
from macros import *
from torch.nn import functional as F
import utils
from torch.nn.utils.rnn import pack_padded_sequence as pack, \
    pad_packed_sequence as unpack

class NTMMemory(nn.Module):

    def __init__(self, N, M):
        super(NTMMemory, self).__init__()
        self.N = N
        self.M = M
        self.mem_bias = nn.Parameter(torch.Tensor(N, M),
                                     requires_grad=False)

    def reset(self, bsz):
        self.bsz = bsz
        self.memory = self.mem_bias.expand(bsz, self.N, self.M)

    def size(self):
        return self.N, self.M

    def read(self, w):
        return torch.matmul(w.unsqueeze(1), self.memory).squeeze(1)

    def write(self, w, e, a):
        self.pre_mem = self.memory
        erase = torch.matmul(w.unsqueeze(-1), e.unsqueeze(1))
        add = torch.matmul(w.unsqueeze(-1), a.unsqueeze(1))
        self.memory = self.pre_mem * (1 - erase) + add

    def _similarity(self, k, beta):
        k = k.view(self.bsz, 1 ,-1)
        w = F.softmax(beta *
                      F.cosine_similarity(self.memory + 1e-16,
                                        k + 1e-16, dim=-1),
                      dim=1)
        return w

    def _interpolate(self, w_prev, wc, g):
        return g * wc + (1 - g) * w_prev

    def _shift(self, wg, s):
        res = []
        for b in range(self.bsz):
            res.append(utils.
                       modulo_convolve(wg[b], s[b]).
                       unsqueeze(0))

        res = torch.cat(res, dim=0)
        return res

    def _sharpen(self, ww, gamma):
        w = ww ** gamma
        w = torch.div(w,
                      torch.sum(w, dim=1).view(-1, 1) + 1e-16)
        return w

    def _preproc(self, k, beta, g, s, gamma):
        k = k.clone()
        beta = F.softplus(beta)
        g = torch.sigmoid(g)
        s = F.softmax(s, dim=1)
        gamma = 1 + F.softplus(gamma)

        return k, beta, g, s, gamma

    def address(self, k, beta, g, s, gamma, w_pre):
        k, beta, g, s, gamma = \
            self._preproc(k, beta, g, s, gamma)

        wc = self._similarity(k, beta)
        wg = self._interpolate(w_pre, wc, g)
        ww = self._shift(wg, s)
        w = self._sharpen(ww, gamma)
        return w

class NTMReadHead(nn.Module):

    def __init__(self, memory, cdim):
        super(NTMReadHead, self).__init__()
        N, M = memory.size()
        self.memory = memory
        self.N = N
        self.M = M
        self.read_lens = [M, 1, 1, 3, 1]
        self.fc_read = nn.Linear(cdim, sum(self.read_lens))
        self.init_state = nn.Parameter(torch.zeros(N),
                                       requires_grad=False)

    def create_new_state(self, bsz):
        return self.init_state.expand(bsz, self.N)

    def is_read_head(self):
        return True

    def forward(self, hid, w_pre):
        o = self.fc_read(hid)
        k, beta, g, s, gamma = \
            utils.split_cols(o, self.read_lens)
        w = self.memory.address(k, beta, g, s, gamma, w_pre)
        r = self.memory.read(w)

        return r, w

class NTMWriteHead(nn.Module):

    def __init__(self, memory, cdim):
        super(NTMWriteHead, self).__init__()
        N, M = memory.size()
        self.memory = memory
        self.N = N
        self.M = M
        self.write_lens = [M, 1, 1, 3, 1, M, M]
        self.fc_write = nn.Linear(cdim, sum(self.write_lens))
        self.init_state = nn.Parameter(torch.zeros(N),
                                       requires_grad=False)

    def create_new_state(self, bsz):
        return self.init_state.expand(bsz, self.N)

    def is_read_head(self):
        return False

    def forward(self, hid, w_pre):
        o = self.fc_write(hid)
        k, beta, g, s, gamma, e, a = \
            utils.split_cols(o, self.write_lens)

        e = torch.sigmoid(e)
        w = self.memory.address(k, beta, g, s, gamma, w_pre)

        self.memory.write(w, e, a)
        return w


class EncoderNTM(nn.Module):
    def __init__(self,
                 idim,
                 cdim,
                 num_heads,
                 N,
                 M):
        super(EncoderNTM, self).__init__()

        self.idim = idim
        self.hdim = cdim + num_heads * M
        # controller hidden dimension
        self.cdim = cdim
        self.num_heads = num_heads
        self.N = N
        self.M = M

        self.memory = NTMMemory(N, M)
        self.controller = nn.LSTM(idim + M * num_heads, cdim)

        self.heads = nn.ModuleList([])
        for _ in range(num_heads):
            self.heads += [
                NTMReadHead(self.memory, cdim),
                NTMWriteHead(self.memory, cdim)
            ]

        self.h0 = nn.Parameter(torch.randn(cdim) * 0.05, requires_grad=True)
        self.c0 = nn.Parameter(torch.randn(cdim) * 0.05, requires_grad=True)
        self.r0 = [nn.Parameter(torch.randn(1, M) * 0.02, requires_grad=False)
                   for _ in range(num_heads)]

        for i, r in enumerate(self.r0):
            setattr(self, 'r0_%d' % i, r)

    def forward(self, input):
        embs = input['embs']
        lens = input['lens']

        bsz = embs.shape[1]
        self.memory.reset(bsz)
        reads = [r.clone().repeat(bsz, 1) for r in self.r0]
        head_states = [head.create_new_state(bsz) for head in self.heads]
        h = self.h0.expand(1, bsz, self.cdim).contiguous()
        c = self.c0.expand(1, bsz, self.cdim).contiguous()

        os = []
        hs = []
        cs = []
        for emb in embs:
            controller_inp = torch.cat([emb] + reads, dim=1).unsqueeze(0)
            controller_outp, (h, c) = self.controller(controller_inp, (h, c))
            controller_outp = controller_outp.squeeze(0)

            # order??
            hs.append(h)
            cs.append(c)


            reads = []
            head_states_next = []
            for head, head_state in zip(self.heads, head_states):
                if head.is_read_head():
                    r, head_state = head(controller_outp, head_state)
                    reads += [r]
                else:
                    head_state = head(controller_outp, head_state)

                head_states_next += [head_state]

            o = torch.cat([controller_outp] + reads, dim=1)
            os.append(o.unsqueeze(0))

        os = torch.cat(os, dim=0)
        hs = torch.cat(hs, dim=0)
        cs = torch.cat(cs, dim=0)
        h = torch.cat([hs[lens[b] - 1, b, :].unsqueeze(0) for b in range(bsz)],
                         dim=0).unsqueeze(0)
        c = torch.cat([cs[lens[b] - 1, b, :].unsqueeze(0) for b in range(bsz)],
                         dim=0).unsqueeze(0)

        return {'output': os,
                'hid': (h, c)}