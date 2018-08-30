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
                 nstack,
                 stack_size,
                 sdim,
                 sdepth):
        super(EncoderSARNN, self).__init__()

        # here input dimention is equal to hidden dimention
        self.idim = idim
        self.hdim = hdim
        self.nstack = nstack
        self.ssz = stack_size
        self.sdepth = sdepth
        self.sdim = sdim
        self.nonLinear=nn.Tanh()

        self.hid2hid = nn.Linear(hdim, hdim)
        self.emb2hid = nn.Linear(idim, hdim)

        self.hid2act = nn.Linear(hdim, nstack * len(ACTS))

        self.hid2stack = nn.Linear(hdim, nstack * sdim)
        self.stack2hid = nn.Linear(nstack * sdim * sdepth,
                                   nstack * hdim)

        self.mem_bias = nn.Parameter(torch.Tensor(nstack, sdim),
                                     requires_grad=False)
        stdev = 1 / (np.sqrt(nstack * stack_size + sdim))
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
                     p_noop, push_vals):

        # stack: (bsz, nstack, ssz, sdim)
        # p_push, p_pop, p_noop: (bsz, nstack, 1, 1)
        # push_vals: (bsz, nstack, sdim)
        p_push = p_push.unsqueeze(-1)
        p_pop = p_pop.unsqueeze(-1)
        p_noop = p_noop.unsqueeze(-1)

        stack_push = self.W_push.matmul(stack)
        stack_push[:, :, 0, :] += push_vals

        stack_pop = self.W_pop.matmul(stack)
        # fill the stack with empty elements
        stack_pop[:, :, self.ssz - 1:, :] += \
            self.mem_bias.unsqueeze(1).unsqueeze(0)

        stack  = p_push * stack_push + p_pop * stack_pop + p_noop * stack
        return stack

    def forward(self, input):

        embs = input['embs']
        lens = input['lens']
        bsz = embs.shape[1]

        if 'stack' in input.keys() and \
            input['stack'] is not None:
            stack = input['stack']
        else:
            stack = self.mem_bias.unsqueeze(1). \
                unsqueeze(0). \
                expand(bsz,
                       self.nstack,
                       self.ssz,
                       self.sdim)

        hid = self.zero.expand(bsz, self.hdim)

        stack_res = [None] * bsz
        output = []
        for t, emb in enumerate(embs):

            mhid = self.emb2hid(emb) + self.hid2hid(hid)

            # tops: (bsz, nstack * stack_depth * sdim)
            tops = stack[:, :, :self.sdepth, :].contiguous(). \
                view(bsz, -1)

            # read: (bsz, nstack * hdim)
            read = self.stack2hid(tops)
            read = read.view(-1, self.nstack, self.hdim)
            mhid += read.sum(dim=1)

            # act: (bsz, nstack * nacts)
            act = self.hid2act(hid)
            # act: (bsz, nstack, nacts)
            act = act.view(-1, self.nstack, len(ACTS))
            act = F.softmax(act, dim=-1)
            # p_xxx: (bsz, nstack, 1)
            p_push, p_pop, p_noop = \
                act.chunk(len(ACTS), dim=-1)

            # push_vals: (bsz, nstack * sdim)
            push_vals = self.hid2stack(hid)
            # push_vals: (bsz, nstack, sdim)
            push_vals = push_vals.view(-1, self.nstack, self.sdim)
            push_vals = self.nonLinear(push_vals)
            stack = self.update_stack(stack, p_push, p_pop, p_noop, push_vals)

            hid = self.nonLinear(mhid)
            outp = hid
            output.append(outp.unsqueeze(0))
            for b, l in enumerate(lens):
                if t == l-1:
                    stack_res[b] = stack[b].unsqueeze(0)

        output = torch.cat(output, dim=0)
        stack = torch.cat(stack_res, dim=0)
        hid = torch.cat([output[lens[b] - 1, b, :].unsqueeze(0) for b in range(bsz)],
                        dim=0).unsqueeze(0)

        return {'output': output,
                'hid': hid,
                'stack': stack}

class DecoderSARNN(nn.Module):
    def __init__(self,
                 idim,
                 hdim,
                 nstack,
                 stack_size,
                 sdim,
                 sdepth):
        super(DecoderSARNN, self).__init__()

        # here input dimention is equal to hidden dimention
        self.idim = idim
        self.hdim = hdim
        self.nstack = nstack
        self.ssz = stack_size
        self.sdepth = sdepth
        self.sdim = sdim
        self.nonLinear=nn.Tanh()

        self.hid2hid = nn.Linear(hdim, hdim)
        self.emb2hid = nn.Linear(idim, hdim)

        self.hid2act = nn.Linear(hdim, nstack * len(ACTS))

        self.hid2stack = nn.Linear(hdim, nstack * sdim)
        self.stack2hid = nn.Linear(nstack * sdim * sdepth,
                                   nstack * hdim)

        self.mem_bias = nn.Parameter(torch.Tensor(nstack, sdim),
                                     requires_grad=False)
        stdev = 1 / (np.sqrt(nstack * stack_size + sdim))
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
                     p_noop, push_vals):

        # stack: (bsz, nstack, ssz, sdim)
        # p_push, p_pop, p_noop: (bsz, nstack, 1, 1)
        # push_vals: (bsz, nstack, sdim)
        p_push = p_push.unsqueeze(-1)
        p_pop = p_pop.unsqueeze(-1)
        p_noop = p_noop.unsqueeze(-1)

        stack_push = self.W_push.matmul(stack)
        stack_push[:, :, 0, :] += push_vals

        stack_pop = self.W_pop.matmul(stack)
        # fill the stack with empty elements
        stack_pop[:, :, self.ssz - 1:, :] += \
            self.mem_bias.unsqueeze(1).unsqueeze(0)

        stack  = p_push * stack_push + p_pop * stack_pop + p_noop * stack
        return stack

    def forward(self, input):

        inp = input['inp']
        hid = input['hid']
        bsz = inp.shape[1]

        if 'stack' in input.keys() and \
            input['stack'] is not None:
            stack = input['stack']
        else:
            stack = self.mem_bias.unsqueeze(1). \
                unsqueeze(0). \
                expand(bsz,
                       self.nstack,
                       self.ssz,
                       self.sdim)

        if type(hid) == tuple:
            hid = hid[0]

        output = []
        for t, emb in enumerate(inp):

            mhid = self.emb2hid(emb) + self.hid2hid(hid)

            # tops: (bsz, nstack * stack_depth * sdim)
            tops = stack[:, :, :self.sdepth, :].contiguous(). \
                view(bsz, -1)

            # read: (bsz, nstack * hdim)
            read = self.stack2hid(tops)
            read = read.view(-1, self.nstack, self.hdim)
            mhid += read.sum(dim=1)

            # act: (bsz, nstack * nacts)
            act = self.hid2act(hid)
            # act: (bsz, nstack, nacts)
            act = act.view(-1, self.nstack, len(ACTS))
            act = F.softmax(act, dim=-1)
            # p_xxx: (bsz, nstack, 1)
            p_push, p_pop, p_noop = \
                act.chunk(len(ACTS), dim=-1)

            # push_vals: (bsz, nstack * sdim)
            push_vals = self.hid2stack(hid)
            # push_vals: (bsz, nstack, sdim)
            push_vals = push_vals.view(-1, self.nstack, self.sdim)
            push_vals = self.nonLinear(push_vals)
            stack = self.update_stack(stack, p_push, p_pop, p_noop, push_vals)

            hid = self.nonLinear(mhid)
            output.append(hid)

        output = torch.cat(output, dim=0)

        return {'output': output,
                'hid': hid,
                'stack': stack}