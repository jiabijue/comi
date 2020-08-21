import os
from collections import Counter, OrderedDict

import torch
import numpy as np

from utils.augment import *


class Vocab(object):
    def __init__(self, special: list, delimiter=None, vocab_file=None):
        self.special = special
        self.delimiter = delimiter
        self.vocab_file = vocab_file

    def tokenize(self, line, add_eos=False):
        line = line.strip()

        # empty delimiter '' will evaluate False
        if self.delimiter == '':
            symbols = line
        else:
            symbols = line.split(self.delimiter)

        if add_eos:
            return symbols + ['<eos>']
        else:
            return symbols

    def _build_from_file(self, vocab_file):
        self.idx2sym = []
        self.sym2idx = OrderedDict()

        with open(vocab_file, 'r', encoding='utf-8') as f:
            for line in f:
                symb = line.strip().split(',')[-1]
                self.add_symbol(symb)

    def build_vocab(self):
        if self.vocab_file:
            print('building vocab from {}'.format(self.vocab_file))
            self._build_from_file(self.vocab_file)
            print('final vocab size {}'.format(len(self)))
        else:
            print('You need to provide a vocab.txt')

    def encode_file(self, path, ordered=False, verbose=False, add_eos=False,
                    augment_instrument=False, augment_pitch=False, augment_duration=False, augment_velocity=False):
        if verbose: print('encoding file {}'.format(path))
        assert os.path.exists(path)
        encoded = []
        with open(path, 'r', encoding='utf-8') as f:
            tokens = f.read().strip().splitlines()

            if augment_instrument:
                if np.random.rand() > 0.5:
                    tokens = instrument_augment(tokens)

            if augment_pitch:
                transpose_amt = int(np.random.randint(-6, 7))
                tokens = pitch_transpose(tokens, transpose_amt)

            if augment_duration:
                playback_speed = np.random.uniform(low=0.95, high=1.05)
                tokens = time_stretch(tokens, playback_speed)

            if augment_velocity:
                augment_value = int(np.random.randint(-20, 21))
                tokens = velocity_augment(tokens, augment_value)

            collapsed = [' '.join(tokens)]
            for line in collapsed:
                symbols = self.tokenize(line, add_eos=add_eos)
                encoded.append(self.convert_to_tensor(symbols))

        if ordered:
            encoded = torch.cat(encoded)

        return encoded

    def add_special(self, sym):
        if sym not in self.sym2idx:
            self.idx2sym.append(sym)
            self.sym2idx[sym] = len(self.idx2sym) - 1
            setattr(self, '{}_idx'.format(sym.strip('<>')), self.sym2idx[sym])

    def add_symbol(self, sym):
        if sym not in self.sym2idx:
            self.idx2sym.append(sym)
            self.sym2idx[sym] = len(self.idx2sym) - 1

    def get_idx(self, sym):
        return self.sym2idx[sym]

    def get_indices(self, symbols):
        return [self.get_idx(sym) for sym in symbols]

    def convert_to_tensor(self, symbols):
        return torch.LongTensor(self.get_indices(symbols))

    def __len__(self):
        return len(self.idx2sym)
