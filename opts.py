import argparse
import os
from macros import *

def model_opts(parser):
    group = parser.add_argument_group('model')
    group.add_argument('-edim', type=int, default=200)
    group.add_argument('-hdim', type=int, default=200)
    group.add_argument('-sdim', type=int, default=50)
    group.add_argument('-stack_size', type=int, default=100)
    group.add_argument('-stack_depth', type=int, default=2)
    group.add_argument('-fine_tuning', default=False, action='store_true')
    group.add_argument('-emb_type', type=str, default='one-hot')
    # group.add_argument('-enc_type', type=str, default='lstm')
    group.add_argument('-enc_type', type=str, default='simp-rnn')
    # group.add_argument('-dec_type', type=str, default='simp-rnn')
    group.add_argument('-dec_type', type=str, default='lstm')

def train_opts(parser):
    group = parser.add_argument_group('train')
    group.add_argument('-ftrain', type=str, default=os.path.join(SCAN, 'tasks_train_simple.txt'))
    group.add_argument('-fvalid', type=str, default=os.path.join(SCAN, 'tasks_test_simple.txt'))
    # group.add_argument('-ftrain', type=str, default=os.path.join(PROP_ENTAIL, 'train.txt'))
    # group.add_argument('-fvalid', type=str, default=os.path.join(PROP_ENTAIL, 'validate.txt'))
    # group.add_argument('-ftrain', type=str, default=os.path.join(RTE, 'train.tsv'))
    # group.add_argument('-fvalid', type=str, default=os.path.join(RTE, 'dev.tsv'))
    group.add_argument('-load_idx', type=int, default=-1)
    group.add_argument('-bsz', type=int, default=64)
    group.add_argument('-min_freq', type=int, default=2)
    group.add_argument('-nepoch', type=int, default=10)
    group.add_argument('-save_per', type=int, default=2)
    # group.add_argument('-task', type=str, default='prop-entail')
    group.add_argument('-task', type=str, default='scan')
    group.add_argument('-gpu', type=int, default=-1)
    group.add_argument('-lr', type=float, default=1e-3)
    group.add_argument('-lm_coef', type=float, default=0.5)
    group.add_argument('-wdecay', type=float, default=0)

