import numpy as np
from torch import nn
import torch
from macros import *
from torch.nn import functional as F
import utils
from torch.nn.utils.rnn import pack_padded_sequence as pack, \
    pad_packed_sequence as unpack

class EncoderLSTM(nn.Module):
    def __init__(self, idim, hdim, dropout):
        super(EncoderLSTM, self).__init__()
        self.idim = idim
        self.hdim = hdim
        self.lstm = nn.LSTM(input_size=idim,
                            hidden_size=hdim,
                            dropout = dropout)
        self.h0 = nn.Parameter(torch.zeros(hdim), requires_grad=False)
        self.c0 = nn.Parameter(torch.zeros(hdim), requires_grad=False)

    def forward(self, input):
        embs = input['embs']
        lens = input['lens']

        lens_sorted, perm = lens.sort(0, descending=True)
        perm_back = [0] * len(perm)
        for i, idx in enumerate(perm):
            perm_back[idx] = i

        embs_sorted = embs[:,perm,:]

        embs_packed = pack(embs_sorted, list(lens_sorted.data), batch_first=False)
        bsz = embs_sorted.shape[1]
        h0 = self.h0.expand(1, bsz, self.hdim).contiguous()
        c0 = self.c0.expand(1, bsz, self.hdim).contiguous()

        output_sorted, hid_sorted = self.lstm(embs_packed, (h0, c0))
        output_sorted = unpack(output_sorted, batch_first=False)[0]

        output = output_sorted[:, perm_back, :]
        hid = (hid_sorted[0][:, perm_back, :],
               hid_sorted[1][:, perm_back, :])

        return {'output': output,
                'hid': hid}

class DecoderLSTM(nn.Module):
    def __init__(self, idim, hdim, dropout):
        super(DecoderLSTM, self).__init__()
        self.idim = idim
        self.hdim = hdim
        self.rnn = nn.LSTM(input_size=idim,
                           hidden_size=hdim,
                           dropout=dropout)
        self.c0 = nn.Parameter(torch.zeros(hdim),
                               requires_grad=False)

    def forward(self, input):
        inp = input['inp']
        hid = input['hid']

        bsz = inp.shape[1]
        if type(hid) != tuple:
            c0 = self.c0.expand(1, bsz, self.hdim).\
                contiguous()
            hid = (hid, c0)

        output, hid = self.rnn(inp, hid)
        return {'output': output,
                'hid': hid}

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