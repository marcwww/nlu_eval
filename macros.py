import os
ACTS = {'push':0,
        'pop':1,
        # 'noop':2
        }
PAD = '<pad>'
UNK = '<unk>'
EOS = '<eos>'
SOS = '<sos>'
SEP = '<sep>'
DATA = 'data/'
RTE = os.path.join(DATA, 'RTE')
RES = 'res/'
PROP_ENTAIL = os.path.join(DATA, 'prop-entail')
SCAN = os.path.join(DATA, 'scan')