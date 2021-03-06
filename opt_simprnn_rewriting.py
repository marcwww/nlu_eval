import argparse
import os
from macros import *

def model_opts(parser):
    group = parser.add_argument_group('model')
    group.add_argument('-edim', type=int, default=256)
    # group.add_argument('-edim', type=int, default=32)
    # group.add_argument('-edim', type=int, default=50)
    # group.add_argument('-edim', type=int, default=8 + 1)
    # group.add_argument('-edim', type=int, default=8 + 2)
    # group.add_argument('-edim', type=int, default=6 + 1)
    # group.add_argument('-hdim', type=int, default=256)
    group.add_argument('-hdim', type=int, default=120)
    # group.add_argument('-hdim', type=int, default=32)
    # group.add_argument('-hdim', type=int, default=50)
    group.add_argument('-odim', type=int, default=8)
    # group.add_argument('-odim', type=int, default=8 + 1)
    # group.add_argument('-odim', type=int, default=6)
    # group.add_argument('-edim', type=int, default=32)
    # group.add_argument('-hdim', type=int, default=32)
    # group.add_argument('-sdim', type=int, default=36)
    # group.add_argument('-sdim', type=int, default=120)
    group.add_argument('-sdim', type=int, default=120)
    group.add_argument('-nstack', type=int, default=2)
    group.add_argument('-stack_size', type=int, default=50)
    group.add_argument('-stack_depth', type=int, default=2)
    group.add_argument('-fine_tuning', default=False, action='store_true')
    # group.add_argument('-dropout', type=float, default=0)
    group.add_argument('-dropout', type=float, default=0.1)
    group.add_argument('-fix_emb', default=False, action='store_true')
    group.add_argument('-emb_type', type=str, default='one-hot')
    # group.add_argument('-emb_type', type=str, default='dense')
    group.add_argument('-enc_type', type=str, default='simp-rnn')
    # group.add_argument('-enc_type', type=str, default='lstm')
    # group.add_argument('-enc_type', type=str, default='alstm')
    # group.add_argument('-enc_type', type=str, default='ntm')
    # group.add_argument('-enc_type', type=str, default='nse')
    # group.add_argument('-enc_type', type=str, default='sarnn')
    # group.add_argument('-enc_type', type=str, default='tardis')
    # group.add_argument('-soft_enc', default=False, action='store_true')
    group.add_argument('-soft_enc', default=True, action='store_true')
    # group.add_argument('-num_heads', type=int, default=4)
    group.add_argument('-num_heads', type=int, default=1)
    # group.add_argument('-N', type=int, default=40)
    group.add_argument('-N', type=int, default=100)
    # group.add_argument('-N', type=int, default=40)
    group.add_argument('-a_ratio', type=float, default=12/120)
    # group.add_argument('-a_ratio', type=float, default=4/36)
    # group.add_argument('-a_ratio', type=float, default=2/10)
    # group.add_argument('-M', type=int, default=36)
    group.add_argument('-M', type=int, default=120)
    # group.add_argument('-M', type=int, default=10)
    # group.add_argument('-soft_dec', default=False, action='store_true')
    group.add_argument('-soft_dec', default=True, action='store_true')
    group.add_argument('-dec_type', type=str, default='simp-rnn')
    # group.add_argument('-dec_type', type=str, default='lstm')
    # group.add_argument('-dec_type', type=str, default='alstm')
    # group.add_argument('-dec_type', type=str, default='ntm')
    # group.add_argument('-dec_type', type=str, default='nse')
    # group.add_argument('-dec_type', type=str, default='sarnn')
    # group.add_argument('-dec_type', type=str, default='tardis')

def train_opts(parser):
    group = parser.add_argument_group('train')
    group.add_argument('-seed', type=int, default=1000)
    # group.add_argument('-ftrain', type=str, default=os.path.join(SCAN, 'tasks_train_simple.txt'))
    # group.add_argument('-fvalid', type=str, default=os.path.join(SCAN, 'tasks_test_simple.txt'))
    # group.add_argument('-ftrain', type=str, default=os.path.join(SCAN, 'tasks_train_addprim_turn_left.txt'))
    # group.add_argument('-fvalid', type=str, default=os.path.join(SCAN, 'tasks_test_addprim_turn_left.txt'))
    # group.add_argument('-ftrain', type=str, default=os.path.join(SCAN, 'tasks_train_addprim_jump.txt'))
    # group.add_argument('-fvalid', type=str, default=os.path.join(SCAN, 'tasks_test_addprim_jump.txt'))
    # group.add_argument('-ftrain', type=str, default=os.path.join(SCAN, 'tasks_train_length.txt'))
    # group.add_argument('-fvalid', type=str, default=os.path.join(SCAN, 'tasks_test_length.txt'))
    group.add_argument('-ftrain', type=str, default=os.path.join(REWRITING, 'grammar_std.train.full.tsv'))
    # group.add_argument('-fvalid', type=str, default=os.path.join(REWRITING, 'grammar.val.tsv'))
    group.add_argument('-fvalid', type=str, default=os.path.join(REWRITING, 'grammar_std.tst.full.tsv'))
    # group.add_argument('-ftrain', type=str, default=os.path.join(PROP_ENTAIL, 'train.txt'))
    # group.add_argument('-fvalid', type=str, default=os.path.join(PROP_ENTAIL, 'validate.txt'))
    # group.add_argument('-fvalid', type=str, default=os.path.join(PROP_ENTAIL, 'test_hard.txt'))
    # group.add_argument('-ftrain', type=str, default=os.path.join(PATTERN, 'copy_train1-10.pkl'))
    # group.add_argument('-fvalid', type=str, default=os.path.join(PATTERN, 'copy_valid11-20.pkl'))
    # group.add_argument('-ftrain', type=str, default=os.path.join(PATTERN, 'copy_train1-5.pkl'))
    # group.add_argument('-fvalid', type=str, default=os.path.join(PATTERN, 'copy_valid1-5.pkl'))
    # group.add_argument('-fvalid', type=str, default=os.path.join(PATTERN, 'copy_valid6-10.pkl'))
    # group.add_argument('-ftrain', type=str, default=os.path.join(RTE, 'train.tsv'))
    # group.add_argument('-fvalid', type=str, default=os.path.join(RTE, 'dev.tsv'))
    group.add_argument('-fload', type=str, default=None)
    # group.add_argument('-fload', type=str, default='rewriting-simple-sarnn-sarnn-1536563947.model')
    # group.add_argument('-fload', type=str, default='rewriting-simple-sarnn-sarnn-1536483321.model')
    # group.add_argument('-fload', type=str, default='rewriting-simple-sarnn-sarnn-1536461801.model')
    # group.add_argument('-fload', type=str, default='rewriting-simple-tardis-tardis-1536459710.model')
    # group.add_argument('-fload', type=str, default='rewriting-simple-simp-rnn-simp-rnn-1536337115.model')
    # group.add_argument('-fload', type=str, default='rewriting-simple-lstm-lstm-1536337152.model')
    # group.add_argument('-fload', type=str, default='rewriting-simple-sarnn-sarnn-1536337171.model')
    # group.add_argument('-fload', type=str, default='rewriting-simple-tardis-tardis-1536337166.model')
    # group.add_argument('-fload', type=str, default='prop-entail-2enc-simple-alstm-1535938865.model')
    # group.add_argument('-fload', type=str, default='prop-entail-2enc-simple-sarnn-1535938958.model')
    # group.add_argument('-fload', type=str, default='prop-entail-2enc-simple-lstm-1535938849.model')
    # group.add_argument('-fload', type=str, default='prop-entail-2enc-simple-ntm-1536024262.model')
    # group.add_argument('-fload', type=str, default='prop-entail-2enc-simple-nse-1535938935.model')
    # group.add_argument('-fload', type=str, default='prop-entail-2enc-simple-simp-rnn-1535938838.model')
    # group.add_argument('-fload', type=str, default='prop-entail-2enc-hard-tardis-1536115286.model')
    group.add_argument('-bsz', type=int, default=64)
    # group.add_argument('-bsz', type=int, default=4)
    group.add_argument('-min_freq', type=int, default=0)
    group.add_argument('-nepoch', type=int, default=30)
    group.add_argument('-save_per', type=int, default=2)
    # group.add_argument('-task', type=str, default='pattern')
    group.add_argument('-task', type=str, default='rewriting')
    # group.add_argument('-task', type=str, default='prop-entail')
    # group.add_argument('-task', type=str, default='prop-entail-2enc')
    # group.add_argument('-task', type=str, default='scan')
    group.add_argument('-gpu', type=int, default=-1)
    group.add_argument('-lr', type=float, default=1e-3)
    # group.add_argument('-lr', type=float, default=1e-2)
    # group.add_argument('-lr', type=float, default=1e-4)
    # group.add_argument('-lr', type=float, default=1)
    # group.add_argument('-lr', type=float, default=0.125)
    group.add_argument('-lm_coef', type=float, default=0.5)
    group.add_argument('-wdecay', type=float, default=0)

    # for copy-task
    group.add_argument('-seq_width', type=int, default=8)
    # group.add_argument('-sub_task', type=str, default='addprim_turn_left')
    # group.add_argument('-sub_task', type=str, default='addprim_jump')
    # group.add_argument('-sub_task', type=str, default='length')
    # group.add_argument('-sub_task', type=str, default='hard')
    group.add_argument('-sub_task', type=str, default='simple')
    # group.add_argument('-sub_task', type=str, default='copy')
    # group.add_argument('-sub_task', type=str, default='mirror')
    # group.add_argument('-sub_task', type=str, default='repeat')
    group.add_argument('-num_batches_train', type=int, default=5000)
    group.add_argument('-num_batches_valid', type=int, default=1000)
    # group.add_argument('-num_batches_train', type=int, default=500)
    # group.add_argument('-num_batches_valid', type=int, default=10)
    group.add_argument('-min_len_train', type=int, default=1)
    group.add_argument('-max_len_train', type=int, default=10)
    group.add_argument('-repeat_min_train', type=int, default=1)
    group.add_argument('-repeat_max_train', type=int, default=3)
    group.add_argument('-min_len_valid', type=int, default=1)
    group.add_argument('-max_len_valid', type=int, default=20)
    # group.add_argument('-min_len_valid', type=int, default=6)
    # group.add_argument('-max_len_valid', type=int, default=10)
    group.add_argument('-repeat_min_valid', type=int, default=1)
    group.add_argument('-repeat_max_valid', type=int, default=3)


