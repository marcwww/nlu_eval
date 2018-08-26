import numpy as np
from torch.nn.init import xavier_uniform_
import torch
from torch import nn
from torch.nn import functional as F
import logging
import random
import time

LOGGER = logging.getLogger(__name__)


def modulo_convolve(w, s):
    # w: (N)
    # s: (3)
    assert s.size(0) == 3
    t = torch.cat([w[-1:], w, w[:1]], dim=-1)
    c = F.conv1d(t.view(1, 1, -1), s.view(1, 1, -1)).view(-1)
    return c

def split_cols(mat, lengths):
    """Split a 2D matrix to variable length columns."""
    assert mat.size()[1] == sum(lengths), "Lengths must be summed to num columns"
    l = np.cumsum([0] + lengths)
    results = []
    for s, e in zip(l[:-1], l[1:]):
        results += [mat[:, s:e]]
    return results

def one_hot_matrix(stoi, device, edim):

    assert len(stoi) <= edim, \
        'embedding dimension must be larger than voc_size'

    voc_size = len(stoi)
    res = torch.zeros(voc_size,
                      edim,
                      requires_grad=False)
    for i in range(voc_size):
        res[i][i] = 1

    return res.to(device)

def shift_matrix(n):
    W_up = np.eye(n)
    for i in range(n-1):
        W_up[i,:] = W_up[i+1,:]
    W_up[n-1,:] *= 0
    W_down = np.eye(n)
    for i in range(n-1,0,-1):
        W_down[i,:] = W_down[i-1,:]
    W_down[0,:] *= 0
    return W_up,W_down

def avg_vector(i, n):
    V = np.zeros(n)
    V[:i+1] = 1/(i+1)
    return V

def init_model(model):
    for p in model.parameters():
        if p.dim() > 1 and p.requires_grad:
            xavier_uniform_(p)

def progress_bar(percent, loss, epoch):
    """Prints the progress until the next report."""

    fill = int(percent * 40)
    str_disp = "\r[%s%s]: %.2f/epoch %d" % ('=' * fill,
                                         ' ' * (40 - fill),
                                         percent,
                                         epoch)
    for k, v in loss.items():
        str_disp += ' (%s:%.4f)' % (k, v)

    print(str_disp, end='')

def seq_lens(seq, padding_idx):
    mask = seq.data.eq(padding_idx)
    len_total, bsz = seq.shape
    lens = len_total - mask.sum(dim=0)
    return lens

class Attention(nn.Module):
    def __init__(self, hdim):
        super(Attention, self).__init__()
        self.hc2ha = nn.Sequential(nn.Linear(hdim * 2, hdim, bias=False),
                                  nn.Tanh())

    def forward(self, h, enc_outputs):
        # h: (1, bsz, hdim)
        # h_current: (1, bsz, 1, hdim)
        h_current = h.unsqueeze(2)
        # enc_outputs: (len_total, bsz, hdim, 1)
        enc_outputs = enc_outputs.unsqueeze(-1)
        # a: (len_total, bsz, 1, 1)
        a = h_current.matmul(enc_outputs)
        a = F.softmax(a, dim=0)
        # c: (len_total, bsz, hdim, 1)
        c = a * enc_outputs
        # c: (bsz, hdim)
        c = c.sum(0).squeeze(-1).unsqueeze(0)
        ha = self.hc2ha(torch.cat([h, c], dim=-1))
        return ha

def get_ms():
    """Returns the current time in miliseconds."""
    return time.time() * 1000

def init_seed(seed=None):
    """Seed the RNGs for predicatability/reproduction purposes."""
    if seed is None:
        seed = int(get_ms() // 1000)

    LOGGER.info("Using seed=%d", seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    random.seed(seed)

if __name__ == '__main__':
    up, down = shift_matrix(3)
    x = np.array([[0,1,2]]).transpose()
    print(x)
    print(up.dot(x))
    print(down)