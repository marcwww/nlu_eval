import numpy as np
from torch import nn
import torch
from macros import *
from torch.nn import functional as F
import utils
from torch.nn.utils.rnn import pack_padded_sequence as pack, \
    pad_packed_sequence as unpack

class EncoderNSE(nn.Module):
    def __init__(self, idim, dropout):
        super(EncoderNSE, self).__init__()
        self.idim = idim
        self.hdim = idim
        self.lstm_r = nn.LSTM(input_size=idim,
                            hidden_size=idim,
                            dropout = dropout)
        self.lstm_w = nn.LSTM(input_size=idim,
                              hidden_size=idim,
                              dropout=dropout)
        self.compose = nn.Linear(2 * idim, idim)
        self.h0_r = nn.Parameter(torch.zeros(idim), requires_grad=False)
        self.c0_r = nn.Parameter(torch.zeros(idim), requires_grad=False)
        self.h0_w = nn.Parameter(torch.zeros(idim), requires_grad=False)
        self.c0_w = nn.Parameter(torch.zeros(idim), requires_grad=False)

    def forward(self, input):
        # embs: (seq_len, bsz, edim)
        embs = input['embs']
        lens = input['lens']
        bsz = embs.shape[1]

        # mem: (bsz, edim, seq_len)
        if 'nse_states' in input.keys() and \
                input['nse_states'] is not None:
            mem, hid_r = input['nse_states']
            h_r, c_r = hid_r
        else:
            mem = embs.clone().transpose(0, -1).transpose(0, 1)
            h_r = self.h0_r.expand(1, bsz, self.hdim).contiguous()
            c_r = self.c0_r.expand(1, bsz, self.hdim).contiguous()

        h_w = self.h0_w.expand(1, bsz, self.hdim).contiguous()
        c_w = self.c0_w.expand(1, bsz, self.hdim).contiguous()

        h_r_res = [None] * bsz
        c_r_res = [None] * bsz
        h_w_res = [None] * bsz
        c_w_res = [None] * bsz
        mem_res = [None] * bsz

        output = []
        for t, emb in enumerate(embs):
            # o: (1, bsz, edim)
            o, (h_r, c_r) = self.lstm_r(emb.unsqueeze(0), (h_r, c_r))
            # z: (bsz, 1, seq_len)
            z = F.softmax(o.transpose(0, 1).matmul(mem), dim=-1)
            # m: (bsz, edim, 1)
            m = mem.matmul(z.transpose(1, 2))
            # c: (bsz, edim)
            c = self.compose(torch.cat([o.squeeze(0).unsqueeze(-1), m], dim=-1).\
                view(-1, 2 * self.hdim))
            # h: (bsz, edim)
            h, (h_w, c_w) = self.lstm_w(c.unsqueeze(0), (h_w, c_w))

            mem = mem * (1 - z)
            mem = mem + h.squeeze(0).unsqueeze(-1).matmul(z)

            output.append(h)
            for b, l in enumerate(lens):
                if t == l-1:
                    h_r_res[b] = h_r[:,b]
                    c_r_res[b] = c_r[:,b]
                    h_w_res[b] = h_w[:,b]
                    c_w_res[b] = c_w[:,b]
                    mem_res[b] = mem[b].unsqueeze(0)

        output = torch.cat(output, dim=0)
        h_r = torch.cat(h_r_res, dim=0).unsqueeze(0)
        c_r = torch.cat(c_r_res, dim=0).unsqueeze(0)
        h_w = torch.cat(h_w_res, dim=0).unsqueeze(0)
        c_w = torch.cat(c_w_res, dim=0).unsqueeze(0)
        mem = torch.cat(mem_res, dim=0)
        hid = (h_w, c_w)
        nse_states = (mem, (h_r, c_r))

        return {'output': output,
                'hid': hid,
                'nse_states': nse_states}

class DecoderNSE(nn.Module):
    def __init__(self, idim, N, dropout):
        super(DecoderNSE, self).__init__()
        self.idim = idim
        self.hdim = idim
        self.N = N
        self.lstm_r = nn.LSTM(input_size=idim,
                            hidden_size=idim,
                            dropout = dropout)
        self.lstm_w = nn.LSTM(input_size=idim,
                              hidden_size=idim,
                              dropout=dropout)
        self.compose = nn.Linear(2 * idim, idim)
        self.h0_r = nn.Parameter(torch.zeros(idim), requires_grad=False)
        self.c0_r = nn.Parameter(torch.zeros(idim), requires_grad=False)
        self.h0_w = nn.Parameter(torch.zeros(idim), requires_grad=False)
        self.c0_w = nn.Parameter(torch.zeros(idim), requires_grad=False)
        self.mem_bias = nn.Parameter(torch.Tensor(idim, N),
                                     requires_grad=False)
        stdev = 1 / (np.sqrt(N + idim))
        nn.init.uniform(self.mem_bias, -stdev, stdev)

    def forward(self, input):
        # embs: (seq_len, bsz, edim)
        inp = input['inp']
        hid = input['hid']
        bsz = inp.shape[1]

        if 'nse_states' in input.keys() and \
                input['nse_states'] is not None:
            mem, hid_r = input['nse_states']
        else:
            mem = self.mem_bias.clone().expand(bsz, self.idim, self.N)
            # mem = inp.clone().squeeze(0).unsqueeze(-1).expand(bsz, self.idim, self.N)
            h_r = self.h0_r.expand(1, bsz, self.hdim).contiguous()
            c_r = self.c0_r.expand(1, bsz, self.hdim).contiguous()
            hid_r = (h_r, c_r)

        if type(hid) != tuple:
            c0 = self.c0.expand(1, bsz, self.cdim).contiguous()
            hid = (hid, c0)

        output = []
        # mem: (bsz, edim, seq_len)
        for emb in inp:
            # o: (1, bsz, edim)
            o, hid_r = self.lstm_r(emb.unsqueeze(0), hid_r)
            # z: (bsz, 1, seq_len)
            z = F.softmax(o.transpose(0, 1).matmul(mem), dim=-1)
            # m: (bsz, edim, 1)
            m = mem.matmul(z.transpose(1, 2))
            # c: (bsz, edim)
            c = self.compose(torch.cat([o.squeeze(0).unsqueeze(-1), m], dim=-1).\
                view(-1, 2 * self.hdim))
            # h: (bsz, edim)
            h, hid = self.lstm_w(c.unsqueeze(0), hid)

            mem = mem * (1 - z)
            mem = mem + h.squeeze(0).unsqueeze(-1).matmul(z)

            output.append(h)

        output = torch.cat(output, dim=0)
        nse_states = (mem, hid_r)

        return {'output': output,
                'hid': hid,
                'nse_states': nse_states}
