import argparse
import os
from macros import *

def model_opts(parser):
    group = parser.add_argument_group('model')
    group.add_argument('-edim', type=int, default=100)
    group.add_argument('-hdim', type=int, default=100)
    group.add_argument('-sdim', type=int, default=100)
    group.add_argument('-stack_size', type=int, default=100)
    group.add_argument('-stack_depth', type=int, default=2)

def train_opts(parser):
    group = parser.add_argument_group('train')
    group.add_argument('-ftrain', type=str, default=os.path.join(RTE, 'train.tsv'))
    group.add_argument('-fvalid', type=str, default=os.path.join(RTE, 'dev.tsv'))
    # group.add_argument('-fvalid', type=str, default=os.path.join(RTE, 'test.tsv'))
    # group.add_argument('-bsz', type=int, default=64)
    group.add_argument('-load_idx', type=int, default=-1)
    group.add_argument('-bsz', type=int, default=64)
    group.add_argument('-min_freq', type=int, default=1)
    group.add_argument('-nepoch', type=int, default=10)
    group.add_argument('-save_per', type=int, default=2)
    group.add_argument('-name', type=str, default='nlu_eval')
    group.add_argument('-gpu', type=int, default=-1)
    group.add_argument('-lr', type=float, default=1e-3)
    group.add_argument('-lm_coef', type=float, default=0.5)
    group.add_argument('-wdecay', type=float, default=0)

