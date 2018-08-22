import numpy as np
from torch.nn.init import xavier_uniform_
import torch

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

if __name__ == '__main__':
    up, down = shift_matrix(3)
    x = np.array([[0,1,2]]).transpose()
    print(x)
    print(up.dot(x))
    print(down)