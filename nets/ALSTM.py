import numpy as np
from torch import nn
import torch
from macros import *
from torch.nn import functional as F
import utils
from torch.nn.utils.rnn import pack_padded_sequence as pack, \
    pad_packed_sequence as unpack

class EncoderALSTM(nn.Module):
    def __init__(self, idim, hdim, dropout):
        super(EncoderALSTM, self).__init__()
        self.idim = idim
        self.hdim = hdim
        self.rnn = nn.LSTM(input_size=idim,
                            hidden_size=hdim,
                            dropout = dropout)
        self.h0 = nn.Parameter(torch.zeros(hdim), requires_grad=False)
        self.c0 = nn.Parameter(torch.zeros(hdim), requires_grad=False)
        self.atten = utils.Attention(hdim)

    def forward(self, input):
        embs = input['embs']
        lens = input['lens']
        bsz = embs.shape[1]

        enc_outputs = None
        if 'enc_outputs' in input.keys() and \
            input['enc_outputs'] is not None:
            enc_outputs = input['enc_outputs']

        h = self.h0.expand(1, bsz, self.hdim).contiguous()
        c = self.c0.expand(1, bsz, self.hdim).contiguous()

        h_res = [None] * bsz
        c_res = [None] * bsz
        output = []
        for t, emb in enumerate(embs):

            if enc_outputs is None:
                if len(output) > 0:
                    hids = torch.cat(output, dim=0)
                    h = self.atten(h, hids)
            else:
                h = self.atten(h, enc_outputs)

            o, (h, c) = self.rnn(emb.unsqueeze(0), (h, c))
            output.append(o)
            for b, l in enumerate(lens):
                if t == l-1:
                    h_res[b] = h[:,b].unsqueeze(0)
                    c_res[b] = c[:,b].unsqueeze(0)

        h = torch.cat(h_res, dim=1)
        c = torch.cat(c_res, dim=1)
        output = torch.cat(output, dim=0)

        return {'output': output,
                'hid': (h, c)}

class DecoderALSTM(nn.Module):
    def __init__(self, idim, hdim, dropout):
        super(DecoderALSTM, self).__init__()
        self.idim = idim
        self.hdim = hdim
        self.rnn = nn.LSTM(input_size=idim,
                           hidden_size=hdim,
                           dropout=dropout)
        self.c0 = nn.Parameter(torch.zeros(hdim),
                               requires_grad=False)
        self.atten = utils.Attention(hdim)

    def forward(self, input):
        inp = input['inp']
        hid = input['hid']
        enc_outputs = input['enc_outputs']

        bsz = inp.shape[1]
        if type(hid) != tuple:
            c0 = self.c0.expand(1, bsz, self.hdim).\
                contiguous()
            hid = (hid, c0)

        ha = self.atten(hid[0], enc_outputs)
        hid = (ha, hid[1])

        output, hid = self.rnn(inp, hid)
        return {'output': output,
                'hid': hid}