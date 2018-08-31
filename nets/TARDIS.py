import numpy as np
from torch import nn
import torch
from macros import *
from torch.nn import functional as F
import utils
from torch.nn.utils.rnn import pack_padded_sequence as pack, \
    pad_packed_sequence as unpack
import crash_on_ipy

class EncoderTARDIS(nn.Module):
    def __init__(self, idim, hdim, N, a, c):
        super(EncoderTARDIS, self).__init__()
        self.idim = idim
        self.hdim = hdim
        self.N = N
        self.a = a
        self.c = c

        self.mem_bias = nn.Parameter(torch.zeros(N, a + c),
                                     requires_grad=False)
        stdev = 1 / (np.sqrt(N + a))
        nn.init.uniform(self.mem_bias[:, :a], -stdev, stdev)

        self.h2w = nn.Linear(hdim, a + c, bias=False)
        self.i2w = nn.Linear(idim, a + c, bias=False)
        self.m2w = nn.Linear(a + c, a + c, bias=False)
        self.u2w = nn.Linear(N, a + c, bias=False)

        self.h2gates = nn.Linear(hdim, 3, bias=False)
        self.i2gates = nn.Linear(idim, 3, bias=False)
        self.r2gates = nn.Linear(a + c, 3, bias=False)

        self.h2alpha_beta = nn.Linear(hdim, 2, bias=False)
        self.i2alpha_beta = nn.Linear(idim, 2, bias=False)
        self.r2alpha_beta = nn.Linear(a + c, 2, bias=False)

        self.h2c = nn.Linear(hdim, hdim, bias=False)
        self.i2c = nn.Linear(idim, hdim, bias=False)
        self.r2c = nn.Linear(a + c, hdim, bias=False)

        self.h0 = nn.Parameter(torch.zeros(hdim), requires_grad=False)
        self.c0 = nn.Parameter(torch.zeros(hdim), requires_grad=False)

        self.atten_base = nn.Parameter(torch.Tensor(1, a + c, 1))

        self.h2tau = nn.Linear(hdim, 1)
        self.h2m = nn.Linear(hdim, c, bias=False)

        self.mem = None
        self.u0 = nn.Parameter(torch.zeros(N), requires_grad=False)

    def _read(self, h, inp, u):
        bsz = h.shape[0]
        atten = self.atten_base.expand(bsz, self.a + self.c, 1)
        # mem: (bsz, N, a+c)
        # hid: (bsz, N, a+c)
        hid = self.h2w(h.squeeze(0)).unsqueeze(1) + \
              self.i2w(inp).unsqueeze(1) + \
              self.m2w(self.mem) + \
              self.u2w(u).unsqueeze(1)

        hid = F.tanh(hid)
        # w: (bsz, N, 1)
        w = hid.matmul(atten)

        tau = F.softplus(self.h2tau(h.squeeze(0))) + 1

        w = F.gumbel_softmax(w.squeeze(-1), tau, hard=True)

        # r: (bsz, 1, a+c)
        r = w.unsqueeze(1).matmul(self.mem)
        return r, w

    def _update_hid(self, inp, h_prev, c_prev, r):
        # gates: (bsz, 3)
        gates = self.h2gates(h_prev) + self.i2gates(inp) + self.r2gates(r.squeeze(1))
        gates = F.sigmoid(gates.squeeze(0))
        f, i, o = gates[:, 0], gates[:, 1], gates[:, 2]

        # alpha_beta: (bsz, 2)
        alpha_beta = self.h2alpha_beta(h_prev) + \
                     self.i2alpha_beta(inp) +\
                     self.r2alpha_beta(r.squeeze(1))

        # gumbel-sigmoid??
        alpha_beta = alpha_beta.squeeze(0)
        alpha, beta = alpha_beta[:, 0], alpha_beta[:, 1]

        alpha = utils.gumbel_sigmoid_max(alpha, tau=0.3, hard=True)
        beta = utils.gumbel_sigmoid_max(beta, tau=0.3, hard=True)

        c = beta.unsqueeze(-1) * self.h2c(h_prev) + \
            self.i2c(inp) +\
            alpha.unsqueeze(-1) * self.r2c(r.squeeze(1))
        c = F.tanh(c)
        c = f.unsqueeze(0).unsqueeze(-1) * c_prev + \
            i.unsqueeze(0).unsqueeze(-1) * c

        h = o.unsqueeze(0).unsqueeze(-1) * F.tanh(c)
        return h, c

    def _write(self, w, h, t):
        val = self.h2m(h)
        bsz = self.mem.shape[0]
        if t < self.N:
            pos = [t] * bsz
        else:
            _, pos = torch.topk(w, k=1, dim=-1)
            pos = pos.squeeze(-1)

        # grad???
        # self.mem[range(bsz), pos, self.a:].data = val.squeeze(0)
        # self.mem[range(bsz), pos, self.a:] = val.squeeze(0)
        val = val.squeeze(0)
        # mem: (bsz, N, a+c)
        mem = list(self.mem)
        for b in range(len(mem)):
            a, c = mem[b][pos[b], :self.a], val[b]
            cell = torch.cat([a, c], dim=-1).unsqueeze(0)
            mem[b] = torch.cat([mem[b][:pos[b]],
                                cell,
                                mem[b][pos[b] + 1:]], dim=0).unsqueeze(0)

        self.mem = torch.cat(mem, dim=0)

    def forward(self, input):
        # embs: (seq_len, bsz, edim)
        embs = input['embs']
        lens = input['lens']
        bsz = embs.shape[1]

        if 'tardis_states' in input.keys() and \
            input['tardis_states'] is not None:
            mem, w_sum, T = input['tardis_states']
            self.mem = mem
        else:
            self.mem = self.mem_bias.clone().expand(bsz, self.N, self.a + self.c)
            w_sum = self.u0.expand(bsz, self.N)
            T = 0

        h = self.h0.expand(1, bsz, self.hdim).contiguous()
        c = self.c0.expand(1, bsz, self.hdim).contiguous()

        w_sum_res = [None] * bsz
        mem_res = [None] * bsz
        c_res = [None] * bsz

        output = []
        for t, emb in enumerate(embs):
            # normalize??
            u = F.layer_norm(w_sum, normalized_shape=w_sum.shape[1:])
            r, w = self._read(h, emb, u)
            w_sum = w_sum + w

            h, c = self._update_hid(emb, h, c, r)
            self._write(w, h, t + T)

            output.append(h)
            for b, l in enumerate(lens):
                if t == l-1:
                    w_sum_res[b] = w_sum[b].unsqueeze(0)
                    mem_res[b] = self.mem[b].unsqueeze(0)
                    c_res[b] = c[:, b].unsqueeze(1)

        output = torch.cat(output, dim=0)
        h = torch.cat([output[lens[b] - 1, b, :].unsqueeze(0) for b in range(bsz)],
                        dim=0).unsqueeze(0)

        mem = torch.cat(mem_res, dim=0)
        w_sum = torch.cat(w_sum_res, dim=0)
        c = torch.cat(c_res, dim=1)
        T = T + len(embs)
        tardis_states = (mem, w_sum, T)

        return {'output': output,
                'hid': (h, c),
                'tardis_states': tardis_states}

class DecoderTARDIS(nn.Module):
    def __init__(self, idim, hdim, N, a, c):
        super(DecoderTARDIS, self).__init__()
        self.idim = idim
        self.hdim = hdim
        self.N = N
        self.a = a
        self.c = c

        self.mem_bias = nn.Parameter(torch.zeros(N, a + c),
                                     requires_grad=False)
        stdev = 1 / (np.sqrt(N + a))
        nn.init.uniform(self.mem_bias[:, :a], -stdev, stdev)

        self.h2w = nn.Linear(hdim, a + c, bias=False)
        self.i2w = nn.Linear(idim, a + c, bias=False)
        self.m2w = nn.Linear(a + c, a + c, bias=False)
        self.u2w = nn.Linear(N, a + c, bias=False)

        self.h2gates = nn.Linear(hdim, 3, bias=False)
        self.i2gates = nn.Linear(idim, 3, bias=False)
        self.r2gates = nn.Linear(a + c, 3, bias=False)

        self.h2alpha_beta = nn.Linear(hdim, 2, bias=False)
        self.i2alpha_beta = nn.Linear(idim, 2, bias=False)
        self.r2alpha_beta = nn.Linear(a + c, 2, bias=False)

        self.h2c = nn.Linear(hdim, hdim, bias=False)
        self.i2c = nn.Linear(idim, hdim, bias=False)
        self.r2c = nn.Linear(a + c, hdim, bias=False)

        self.h0 = nn.Parameter(torch.zeros(hdim), requires_grad=False)
        self.c0 = nn.Parameter(torch.zeros(hdim), requires_grad=False)

        self.atten_base = nn.Parameter(torch.Tensor(1, a + c, 1))

        self.h2tau = nn.Linear(hdim, 1)
        self.h2m = nn.Linear(hdim, c, bias=False)

        self.mem = None
        self.u0 = nn.Parameter(torch.zeros(N), requires_grad=False)

    def _read(self, h, inp, u):
        bsz = h.shape[0]
        atten = self.atten_base.expand(bsz, self.a + self.c, 1)
        # mem: (bsz, N, a+c)
        # hid: (bsz, N, a+c)
        hid = self.h2w(h.squeeze(0)).unsqueeze(1) + \
              self.i2w(inp).unsqueeze(1) + \
              self.m2w(self.mem) + \
              self.u2w(u).unsqueeze(1)

        hid = F.tanh(hid)
        # w: (bsz, N, 1)
        w = hid.matmul(atten)

        tau = F.softplus(self.h2tau(h.squeeze(0))) + 1

        # w = F.gumbel_softmax(w.squeeze(-1), tau, hard=False)
        w = F.gumbel_softmax(w.squeeze(-1), tau, hard=True)

        # r: (bsz, 1, a+c)
        r = w.unsqueeze(1).matmul(self.mem)
        return r, w

    def _update_hid(self, inp, h_prev, c_prev, r):
        # gates: (bsz, 3)
        gates = self.h2gates(h_prev) + self.i2gates(inp) + self.r2gates(r.squeeze(1))
        gates = F.sigmoid(gates.squeeze(0))
        f, i, o = gates[:, 0], gates[:, 1], gates[:, 2]

        # alpha_beta: (bsz, 2)
        alpha_beta = self.h2alpha_beta(h_prev) + \
                     self.i2alpha_beta(inp) + \
                     self.r2alpha_beta(r.squeeze(1))

        # gumbel-sigmoid??
        alpha_beta = alpha_beta.squeeze(0)
        alpha, beta = alpha_beta[:, 0], alpha_beta[:, 1]

        alpha = utils.gumbel_sigmoid_max(alpha, tau=0.3, hard=True)
        beta = utils.gumbel_sigmoid_max(beta, tau=0.3, hard=True)
        # alpha = utils.gumbel_sigmoid_max(alpha, tau=0.3, hard=False)
        # beta = utils.gumbel_sigmoid_max(beta, tau=0.3, hard=False)

        c = beta.unsqueeze(-1) * self.h2c(h_prev) + \
            self.i2c(inp) + \
            alpha.unsqueeze(-1) * self.r2c(r.squeeze(1))
        c = F.tanh(c)
        c = f.unsqueeze(0).unsqueeze(-1) * c_prev + \
            i.unsqueeze(0).unsqueeze(-1) * c

        h = o.unsqueeze(0).unsqueeze(-1) * F.tanh(c)
        return h, c

    def _write(self, w, h, t):
        val = self.h2m(h)
        bsz = self.mem.shape[0]
        if t < self.N:
            pos = [t] * bsz
        else:
            _, pos = torch.topk(w, k=1, dim=-1)
            pos = pos.squeeze(-1)

        # grad???
        # self.mem[range(bsz), pos, self.a:].data = val.squeeze(0)
        # self.mem[range(bsz), pos, self.a:] = val.squeeze(0)
        val = val.squeeze(0)
        # mem: (bsz, N, a+c)
        mem = list(self.mem)
        for b in range(len(mem)):
            a, c = mem[b][pos[b], :self.a], val[b]
            cell = torch.cat([a, c], dim=-1).unsqueeze(0)
            mem[b] = torch.cat([mem[b][:pos[b]],
                                cell,
                                mem[b][pos[b] + 1:]], dim=0).unsqueeze(0)

        self.mem = torch.cat(mem, dim=0)

    def forward(self, input):
        # embs: (seq_len, bsz, edim)
        inp = input['inp']
        hid = input['hid']
        bsz = inp.shape[1]

        if 'tardis_states' in input.keys() and \
            input['tardis_states'] is not None:
            mem, w_sum, T = input['tardis_states']
            self.mem = mem
        else:
            self.mem = self.mem_bias.clone().expand(bsz, self.N, self.a + self.c)
            w_sum = self.u0.expand(bsz, self.N)
            T = 0

        if type(hid) != tuple:
            c0 = self.c0.expand(1, bsz, self.hdim).\
                contiguous()
            hid = (hid, c0)

        h, c = hid

        output = []
        for emb in inp:
            # normalize??
            u = F.layer_norm(w_sum, normalized_shape=w_sum.shape[1:])
            r, w = self._read(h, emb, u)
            w_sum = w_sum + w

            h, c = self._update_hid(emb, h, c, r)
            self._write(w, h, T)

            output.append(h)
            T += len(inp)

        output = torch.cat(output, dim=0)
        tardis_states = (self.mem, w_sum, T)

        return {'output': output,
                'hid': (h, c),
                'tardis_states': tardis_states}

