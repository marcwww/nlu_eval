import numpy as np
from torch import nn
import torch
from macros import *
from torch.nn import functional as F
import utils
from torch.nn.utils.rnn import pack_padded_sequence as pack, \
    pad_packed_sequence as unpack

class EncoderSARNN(nn.Module):
    def __init__(self,
                 idim,
                 hdim,
                 stack_size,
                 sdim,
                 stack_depth):
        super(EncoderSARNN, self).__init__()

        # here input dimention is equal to hidden dimention
        self.idim = idim
        self.hdim = hdim
        self.ssz = stack_size
        self.sdepth = stack_depth
        self.sdim = sdim
        self.nonLinear=nn.Tanh()

        self.hid2hid = nn.Linear(hdim, hdim)
        self.emb2hid = nn.Linear(idim, hdim)

        self.hid2act = nn.Linear(hdim, len(ACTS))

        self.hid2stack = nn.Linear(hdim, sdim)
        self.stack2hid = nn.Linear(sdim * self.sdepth, hdim)

        self.mem_bias = nn.Parameter(torch.Tensor(1, sdim),
                                     requires_grad=False)
        stdev = 1 / (np.sqrt(stack_size + sdim))
        nn.init.uniform(self.mem_bias, -stdev, stdev)

        # shift matrix for stack
        W_up, W_down = utils.shift_matrix(stack_size)
        self.W_up = nn.Parameter(torch.Tensor(W_up), requires_grad=False)
        self.W_pop = self.W_up

        self.W_down = nn.Parameter(torch.Tensor(W_down), requires_grad=False)
        self.W_push = self.W_down

        self.zero = nn.Parameter(torch.zeros(1), requires_grad=False)

    def update_stack(self, stack,
                     p_push, p_pop,
                     p_noop, push_val):

        # stack: (bsz, ssz, sdim)
        # p_push, p_pop, p_noop: (bsz, 1)
        # push_val: (bsz, sdim)
        # p_xact: bsz * nstack
        p_push = p_push.unsqueeze(-1)
        p_pop = p_pop.unsqueeze(-1)
        p_noop = p_noop.unsqueeze(-1)

        bsz, ssz, sdim  = stack.shape
        stack_push = self.W_push.matmul(stack)
        stack_push[:, 0, :] += push_val

        stack_pop = self.W_pop.matmul(stack)
        # fill the stack with empty elements
        stack_pop[:, self.ssz - 1:, :] += self.mem_bias

        stack  = p_push * stack_push + p_pop * stack_pop + p_noop * stack
        return stack

    def forward(self, input):

        embs = input['embs']
        lens = input['lens']
        bsz = embs.shape[1]

        hid = self.zero.expand(bsz, self.hdim)
        stack = self.mem_bias.expand(bsz,
                                      self.ssz,
                                      self.sdim)

        output = []
        for emb in embs:

            mhid = self.emb2hid(emb) + self.hid2hid(hid)

            # tops: (bsz, stack_depth * sdim)
            tops = stack[:, :self.sdepth, :].contiguous(). \
                view(bsz,
                     self.sdepth * self.sdim)

            read = self.stack2hid(tops)
            mhid += read

            # act: (bsz, nacts)
            # h_t??
            act = self.hid2act(hid)
            act = F.softmax(act, dim=-1)
            p_push, p_pop, p_noop = \
                act.chunk(len(ACTS), dim=-1)

            push_val = self.hid2stack(hid)
            push_val = self.nonLinear(push_val)
            stack = self.update_stack(stack, p_push, p_pop, p_noop, push_val)

            hid = self.nonLinear(mhid)
            outp = hid
            output.append(outp.unsqueeze(0))

        output = torch.cat(output, dim=0)
        hid = torch.cat([output[lens[b] - 1, b, :].unsqueeze(0) for b in range(bsz)],
                        dim=0).unsqueeze(0)

        return {'output': output,
                'hid': hid}