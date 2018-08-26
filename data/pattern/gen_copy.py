import numpy as np
import torch
import random
from macros import *

def gen(num_batches,
        batch_size,
        seq_width,
        min_len,
        max_len):

    res = []
    num_batches = int(num_batches)
    for batch_num in range(num_batches):
        seq_len = random.randint(min_len, max_len)
        seq = np.random.binomial(1, 0.5, (seq_len, batch_size, seq_width))
        seq = torch.Tensor(seq)

        # The input includes an additional channel used for the delimiter
        inp = torch.zeros(seq_len + 1, batch_size, seq_width + 1)
        inp[:seq_len, :, :seq_width] = seq
        inp[seq_len, :, seq_width] = 1.0  # delimiter in our control channel
        outp = seq.clone()
        res.append((batch_num, inp.float(), outp.float()))

    return res

if __name__ == '__main__':
    # bsz = 2
    # num_batches = 50000
    bsz = 32
    num_batches = 3125
    seq_width = 8

    # for training
    res = gen(num_batches, bsz, seq_width, 1, 10)
    torch.save(res, 'copy_train1-10.pkl')
    # res = torch.load('train1-5.pkl')

    # for validation
    res = gen(num_batches / 50, bsz, seq_width, 1, 10)
    torch.save(res, 'copy_valid1-10.pkl')
    # res = torch.load('valid1-5.pkl')

    res = gen(num_batches / 50, bsz, seq_width, 11, 20)
    torch.save(res, 'copy_valid11-20.pkl')
    # res = torch.load('valid6-10.pkl')
