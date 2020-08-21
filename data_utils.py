import glob
import os

import numpy as np
import torch

from utils.vocabulary import Vocab


class LMOrderedIterator(object):
    def __init__(self, data, bsz, bptt, device='cpu', ext_len=None):
        """
            data -- LongTensor -- the LongTensor is strictly ordered
        """
        self.bsz = bsz
        self.bptt = bptt
        self.ext_len = ext_len if ext_len is not None else 0

        self.device = device

        # Work out how cleanly we can divide the dataset into bsz parts.
        self.n_step = data.size(0) // bsz

        # Trim off any extra elements that wouldn't cleanly fit (remainders).
        data = data.narrow(0, 0, self.n_step * bsz)

        # Evenly divide the data across the bsz batches.
        self.data = data.view(bsz, -1).t().contiguous().to(device)

        # Number of mini-batches
        self.n_batch = (self.n_step + self.bptt - 1) // self.bptt

    def get_batch(self, i, bptt=None):
        if bptt is None: bptt = self.bptt
        seq_len = min(bptt, self.data.size(0) - 1 - i)

        end_idx = i + seq_len
        beg_idx = max(0, i - self.ext_len)

        data = self.data[beg_idx:end_idx]
        target = self.data[i + 1:i + 1 + seq_len]

        return data, target, seq_len

    def get_fixlen_iter(self, start=0):
        for i in range(start, self.data.size(0) - 1, self.bptt):
            yield self.get_batch(i)

    def get_varlen_iter(self, start=0, std=5, min_len=5, max_deviation=3):
        max_len = self.bptt + max_deviation * std
        i = start
        while True:
            bptt = self.bptt if np.random.random() < 0.95 else self.bptt / 2.
            bptt = min(max_len, max(min_len, int(np.random.normal(bptt, std))))
            data, target, seq_len = self.get_batch(i, bptt)
            i += seq_len
            yield data, target, seq_len
            if i >= self.data.size(0) - 2:
                break

    def __iter__(self):
        return self.get_fixlen_iter()


class LMShuffledIterator(object):
    def __init__(self, data, bsz, bptt, device='cpu', ext_len=None, shuffle=False):
        """
            data -- list[LongTensor] -- there is no order among the LongTensors
        """
        self.data = data

        self.bsz = bsz
        self.bptt = bptt
        self.ext_len = ext_len if ext_len is not None else 0

        self.device = device
        self.shuffle = shuffle

    def get_sent_stream(self):
        # index iterator
        epoch_indices = np.random.permutation(len(self.data)) if self.shuffle \
            else np.array(range(len(self.data)))

        # sentence iterator
        for idx in epoch_indices:
            yield self.data[idx]

    def stream_iterator(self, sent_stream):
        # streams for each data in the batch
        streams = [None] * self.bsz

        data = torch.LongTensor(self.bptt, self.bsz)
        target = torch.LongTensor(self.bptt, self.bsz)

        n_retain = 0

        while True:
            # data   : [n_retain+bptt x bsz]
            # target : [bptt x bsz]
            data[n_retain:].fill_(-1)
            target.fill_(-1)

            valid_batch = True

            for i in range(self.bsz):
                n_filled = 0
                try:
                    while n_filled < self.bptt:
                        if streams[i] is None or len(streams[i]) <= 1:
                            streams[i] = next(sent_stream)
                        # number of new tokens to fill in
                        n_new = min(len(streams[i]) - 1, self.bptt - n_filled)
                        # first n_retain tokens are retained from last batch
                        data[n_retain + n_filled:n_retain + n_filled + n_new, i] = \
                            streams[i][:n_new]
                        target[n_filled:n_filled + n_new, i] = \
                            streams[i][1:n_new + 1]
                        streams[i] = streams[i][n_new:]
                        n_filled += n_new
                except StopIteration:
                    valid_batch = False
                    break

            if not valid_batch:
                return

            data = data.to(self.device)
            target = target.to(self.device)

            yield data, target, self.bptt

            n_retain = min(data.size(0), self.ext_len)
            if n_retain > 0:
                data[:n_retain] = data[-n_retain:]
            data.resize_(n_retain + self.bptt, data.size(1))

    def __iter__(self):
        # sent_stream is an iterator
        sent_stream = self.get_sent_stream()

        for batch in self.stream_iterator(sent_stream):
            yield batch


class LMMultiFileIterator(LMShuffledIterator):
    def __init__(self, paths, vocab, bsz, bptt, device='cpu', ext_len=None, shuffle=False,
                 augment_instrument=False, augment_pitch=False, augment_duration=False, augment_velocity=False,
                 skip_short=False, split=None):

        self.paths = paths
        self.vocab = vocab

        self.bsz = bsz
        self.bptt = bptt
        self.ext_len = ext_len if ext_len is not None else 0

        self.device = device
        self.shuffle = shuffle

        self.augment_instrument = augment_instrument
        self.augment_pitch = augment_pitch
        self.augment_duration = augment_duration
        self.augment_velocity = augment_velocity
        self.skip_short = skip_short

        self.split = split  ##

    def get_sent_stream(self, path):
        sents = self.vocab.encode_file(path, add_double_eos=True,
                                       augment_instrument=self.augment_instrument,
                                       augment_pitch=self.augment_pitch,
                                       augment_duration=self.augment_duration,
                                       augment_velocity=self.augment_velocity)
        if self.shuffle:
            np.random.shuffle(sents)
        sent_stream = iter(sents)

        return sent_stream

    def __iter__(self):
        if self.shuffle:
            np.random.shuffle(self.paths)

        sents = []
        print("Encoding {} files ...".format(self.split))
        for path in self.paths:
            sents.extend(self.vocab.encode_file(path,
                                                augment_instrument=self.augment_instrument,
                                                augment_pitch=self.augment_pitch,  # do data augmentation in every epoch
                                                augment_duration=self.augment_duration,
                                                augment_velocity=self.augment_velocity))

        if self.skip_short:
            sents = [s for s in sents if len(s) >= 10]

        if self.shuffle:
            np.random.shuffle(sents)

        sent_stream = iter(sents)
        for batch in self.stream_iterator(sent_stream):
            yield batch


class Corpus(object):
    def __init__(self, path, dataset, *args, **kwargs):
        self.dataset = dataset
        self.vocab = Vocab(*args, **kwargs)

        self.train = glob.glob(os.path.join(path, 'train', '*.txt'))
        self.valid = glob.glob(os.path.join(path, 'valid', '*.txt'))
        self.test = glob.glob(os.path.join(path, 'test', '*.txt'))

        self.vocab.build_vocab()

    def get_iterator(self, split, *args, **kwargs):
        kwargs['split'] = split  ##
        if split == 'train':
            kwargs['shuffle'] = True
            data_iter = LMMultiFileIterator(self.train, self.vocab, *args, **kwargs)

        else:
            data = self.valid if split == 'valid' else self.test
            kwargs['shuffle'] = False
            # I've decided to let this always be true for evaluation
            kwargs['skip_short'] = True
            data_iter = LMMultiFileIterator(data, self.vocab, *args, **kwargs)

        return data_iter


def get_lm_corpus(datadir, dataset):
    fn = './data/{}_corpus_cache.pt'.format(dataset)
    if os.path.exists(fn):
        print('Loading cached dataset...')
        corpus = torch.load(fn)
    else:
        print('Producing dataset {}...'.format(dataset))
        kwargs = {}
        kwargs['special'] = []
        kwargs['vocab_file'] = './data/vocab.txt'

        corpus = Corpus(datadir, dataset, **kwargs)
        torch.save(corpus, fn)

    return corpus


# %% Unit Test
if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='unit test')
    parser.add_argument('--data_corpus', type=str, default='./data/mirex-like/txt_128',
                        help='location of the data corpus')
    parser.add_argument('--dataset', type=str, default='mirex-like',
                        choices=['mirex-like', 'lakh'])
    args = parser.parse_args()

    corpus = get_lm_corpus(args.data_corpus, args.dataset)
    print('Vocab size : {}'.format(len(corpus.vocab.idx2sym)))

    print('-' * 10)
    print('Train iterator')
    for batch in corpus.get_iterator('train', bsz=9, bptt=100,
                                     augment_instrument=True,
                                     augment_pitch=True,
                                     augment_velocity=True):
        print(batch)
        break

    for batch in corpus.get_iterator('valid', bsz=3, bptt=10):
        print(batch)
        break
