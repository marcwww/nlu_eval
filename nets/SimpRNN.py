import numpy as np
from torch import nn
import torch
from macros import *
from torch.nn import functional as F
import utils
from torch.nn.utils.rnn import pack_padded_sequence as pack, \
    pad_packed_sequence as unpack

class EncoderSimpRNN(nn.Module):
    def __init__(self, idim, hdim, dropout):
        super(EncoderSimpRNN, self).__init__()
        self.idim = idim
        self.hdim = hdim
        self.rnn = nn.RNN(input_size=idim,
                          hidden_size=hdim,
                          dropout=dropout)

    def forward(self, input):
        embs = input['embs']
        lens = input['lens']
        bsz = embs.shape[1]
        output, _ = self.rnn(embs)
        hid = torch.cat([output[lens[b] - 1, b, :].unsqueeze(0) for b in range(bsz)],
                         dim=0).unsqueeze(0)

        return {'output': output,
                'hid': hid}

class DecoderSimpRNN(nn.Module):
    def __init__(self, idim, hdim, dropout):
        super(DecoderSimpRNN, self).__init__()
        self.idim = idim
        self.hdim = hdim
        self.rnn = nn.RNN(input_size=idim,
                          hidden_size=hdim,
                          dropout=dropout)

    def forward(self, input, hid):
        if type(hid) == tuple:
            hid = hid[0]

        output, hid = self.rnn(input, hid)
        return {'output': output,
                'hid': hid}