import numpy as np
from torch import nn
import torch
from macros import *
from torch.nn import functional as F
import utils
from torch.nn.utils.rnn import pack_padded_sequence as pack, \
    pad_packed_sequence as unpack

from dnc import DNC

class EncoderDNC(nn.Module):
    def __init__(self,
                 idim,
                 cdim,
                 num_heads,
                 N,
                 M,
                 gpu):
        super(EncoderDNC, self).__init__()

        self.idim = idim
        self.hdim = idim
        self.rnn = DNC(input_size=idim,
                       hidden_size=cdim,
                       nr_cells=N,
                       cell_size=M,
                       read_heads=num_heads,
                       batch_first=False,
                       gpu_id=gpu)

    def forward(self, input):
        embs = input['embs']
        lens = input['lens']
        bsz = embs.shape[1]

        (controller_hidden, memory, read_vectors) = (None, None, None)
        output, (controller_hidden, memory, read_vectors) = \
            self.rnn(embs,
                     (controller_hidden, memory, read_vectors),
                     reset_experience=True)

        hid = torch.cat([output[lens[b] - 1, b, :].unsqueeze(0) for b in range(bsz)],
                        dim=0).unsqueeze(0)

        return {'output': output,
                'hid': hid}