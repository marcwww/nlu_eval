import numpy as np
from torch import nn
import torch
from macros import *
from torch.nn import functional as F
import utils
from torch.nn.utils.rnn import pack_padded_sequence as pack, \
    pad_packed_sequence as unpack

class EncoderSRNN(nn.Module):
    def __init__(self, voc_size, edim, hdim,
                 stack_size, sdim, padding_idx,
                 fine_tuning, stack_depth = 2):
        super(EncoderSRNN, self).__init__()
        # here input dimention is equal to hidden dimention
        self.edim = edim
        self.hdim = hdim
        self.ssz = stack_size
        self.sdepth = stack_depth
        self.sdim = sdim
        self.padding_idx = padding_idx
        self.embedding = nn.Embedding(voc_size,
                                      edim,
                                      padding_idx=padding_idx)
        self.embedding.weight.requires_grad = fine_tuning
        self.nonLinear=nn.ReLU()

        self.hid2hid = nn.Linear(hdim, hdim)
        self.emb2hid = nn.Linear(edim, hdim)

        self.hid2inst = nn.Linear(hdim, len(ACTS) + 1)

        self.hid2stack = nn.Linear(hdim, sdim)
        self.stack2hid = nn.Linear(sdim * self.sdepth, hdim)
        self.stack2u = nn.Linear(sdim * self.sdepth, sdim)

        self.empty_elem = nn.Parameter(torch.Tensor(1, self.sdim))
        # self.empty_elem = nn.Parameter(torch.zeros(1, self.sdim), requires_grad=False)
        # self.empty_elem = nn.Parameter(torch.Tensor(1, self.sdim).uniform_(0, 1e-1), requires_grad=False)

        # shift matrix for stack
        W_up, W_down = utils.shift_matrix(stack_size)
        W_pop = W_up
        for i in range(self.sdepth - 1):
            W_pop = np.matmul(W_pop, W_up)
        self.W_up = nn.Parameter(torch.Tensor(W_up), requires_grad=False)
        self.W_pop = nn.Parameter(torch.Tensor(W_pop), requires_grad=False)

        self.W_down = nn.Parameter(torch.Tensor(W_down), requires_grad=False)
        self.W_push = self.W_down

        self.tau = nn.Parameter(torch.Tensor(1).uniform_(0, 1))
        # self.tau = 1
        self.zero = nn.Parameter(torch.zeros(1), requires_grad=False)
        self.pad = nn.Parameter(torch.LongTensor([padding_idx]), requires_grad=False)

    def update_stack(self, stack,
                     p_push, p_pop,
                     push_val, u_val):

        # stack: (bsz, ssz, sdim)
        # p_push, p_pop, p_noop: (bsz, 1)
        # push_val: (bsz, sdim)
        # p_xact: bsz * nstack
        p_push = p_push.unsqueeze(-1)
        p_pop = p_pop.unsqueeze(-1)

        bsz, ssz, sdim  = stack.shape
        stack_push = self.W_push.matmul(stack)
        stack_push[:, 0, :] += push_val

        stack_pop = self.W_pop.matmul(stack)
        stack_pop = self.W_push.matmul(stack_pop)
        stack_pop[:, 0, :] += u_val
        # fill the stack with empty elements
        stack_pop[:, self.ssz - self.sdepth + 1:, :] += self.empty_elem

        stack  = p_push * stack_push + p_pop * stack_pop
        return stack

    def update_buf(self, buf,
                   p_push, p_pop):

        device = buf.device
        T = buf.shape[1]
        W_up, W_down = utils.shift_matrix(T)
        W_down = torch.Tensor(W_down).to(device)

        # buf: (bsz, T, edim)
        p_push = p_push.unsqueeze(-1)
        p_pop = p_pop.unsqueeze(-1)

        buf_push = buf
        buf_pop = W_down.matmul(buf)
        # buf_noop = buf

        buf = p_push * buf_push + p_pop * buf_pop
        return buf

    def forward(self, inputs):

        bsz = inputs.shape[1]
        hid = self.zero.expand(bsz, self.hdim)
        stack = self.empty_elem.expand(bsz,
                                      self.ssz,
                                      self.sdim)

        # inputs: (N, bsz)
        # stacks: (bsz, ssz, sdim)
        # embs: (length, bsz, edim)
        N = len(inputs)
        T = 2 * N -1
        pads = self.pad.expand(T - N, bsz)
        pads = self.embedding(pads)
        embs = self.embedding(inputs)

        # bufs: (T, bsz, edim)
        buf = torch.cat([embs, pads], dim=0)

        outputs = []
        top_elems = []
        acts = []
        for t in range(T):
            # stack_vals: (bsz, stack_depth * sdim)
            # catenate all the readed vectors:
            tops = stack[:, :self.sdepth, :].contiguous(). \
                view(bsz,
                     self.sdepth * self.sdim)

            input = buf[t]
            # emb: (bsz, embdsz)
            ihid = self.emb2hid(input) + self.stack2hid(tops)
            mhid= self.hid2hid(hid) + ihid
            hid = self.nonLinear(mhid)
            outputs.append(hid.unsqueeze(0))

            # act: (bsz, nacts)
            # act = self.hid2act(hid)
            # inst: (bsz, nacts + 1) probability of actions and gamma
            inst = self.hid2inst(ihid)
            act = inst[:, :len(ACTS)]
            gamma = inst[:, len(ACTS):]

            gamma = 1 + torch.log(1 + torch.exp(gamma))
            act = F.softmax(act, dim=-1)
            act_sharpened = act ** gamma
            act_sharpened = torch.div(act_sharpened, torch.sum(act_sharpened, dim=-1).view(-1, 1) + 1e-16)

            # act = F.gumbel_softmax(act, tau=self.tau)

            # p_push, p_pop, p_noop: (bsz, 1)
            p_push, p_pop = \
                act_sharpened.chunk(len(ACTS), dim=-1)

            buf = self.update_buf(buf.transpose(0, 1), p_push, p_pop).\
                transpose(0, 1)

            # _, act_chosen = torch.topk(act_sharpened, k=1, dim=-1)
            acts.append(act_sharpened.unsqueeze(0))

            # push_val: (bsz, sdim)
            push_val = self.hid2stack(hid)

            # push_val: (bsz, ssz)
            push_val = self.nonLinear(push_val)
            # u_val: (bsz, ssz) unified stack element
            u_val = self.nonLinear(self.stack2u(tops))
            stack = self.update_stack(stack,
                                       p_push, p_pop,
                                       push_val, u_val)
            top_elem = stack[:, 0, :].unsqueeze(0)
            top_elems.append(top_elem)

        outputs = torch.cat(outputs, dim=0)
        # acts: (T, bsz, nacts)
        acts = torch.cat(acts, dim=0)
        # top_elems: (T, bsz, sdim)
        top_elems = torch.cat(top_elems, dim=0)

        return {'outputs':outputs,
                'hid':hid,
                'stack':stack,
                'act':acts,
                'top':top_elems}

class EncoderSimpRNN(nn.Module):
    def __init__(self, idim, hdim):
        super(EncoderSimpRNN, self).__init__()
        self.idim = idim
        self.hdim = hdim
        self.rnn = nn.RNN(input_size=idim,
                          hidden_size=hdim)

    def forward(self, input):
        embs = input['embs']
        lens = input['lens']
        bsz = embs.shape[1]
        output, _ = self.rnn(embs)
        hid = torch.cat([output[lens[b] - 1, b, :].unsqueeze(0) for b in range(bsz)],
                         dim=0).unsqueeze(0)

        return {'output': output,
                'hid': hid}

class EncoderLSTM(nn.Module):
    def __init__(self, idim, hdim):
        super(EncoderLSTM, self).__init__()
        self.idim = idim
        self.hdim = hdim
        self.lstm = nn.LSTM(input_size=idim,
                          hidden_size=hdim)
                            # num_layers=2)
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

class DecoderSimpRNN(nn.Module):
    def __init__(self, idim, hdim):
        super(DecoderSimpRNN, self).__init__()
        self.idim = idim
        self.hdim = hdim
        self.rnn = nn.RNN(input_size=idim,
                          hidden_size=hdim)

    def forward(self, input, hid):
        if type(hid) == tuple:
            hid = hid[0]

        output, hid = self.rnn(input, hid)
        return {'output': output,
                'hid': hid}


class DecoderLSTM(nn.Module):
    def __init__(self, idim, hdim):
        super(DecoderLSTM, self).__init__()
        self.idim = idim
        self.hdim = hdim
        self.rnn = nn.LSTM(input_size=idim,
                          hidden_size=hdim)
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

# LSTM with attention
class DecoderALSTM(nn.Module):
    def __init__(self, idim, hdim):
        super(DecoderALSTM, self).__init__()
        self.idim = idim
        self.hdim = hdim
        self.rnn = nn.LSTM(input_size=idim,
                          hidden_size=hdim)
        self.c0 = nn.Parameter(torch.zeros(hdim),
                               requires_grad=False)
        self.atten = Attention(hdim)

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








