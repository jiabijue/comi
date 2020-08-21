import argparse
import os
import random

import torch

from utils.vocabulary import Vocab
from utils.token_sampler import TokenSampler
from tokenization_by_time import MidiTokenizer


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--model_dir', type=str, help='Directory with model')
    parser.add_argument('--gen_dir', type=str, help='Directory to put generated samples')
    parser.add_argument('--cpu', action='store_false', dest='gpu')
    parser.add_argument('--no_control', action='store_true')
    parser.add_argument('--num', type=int, help='number of samples to generate.')
    parser.add_argument('--mem_len', type=int, help='Max length of Transformer memory')
    parser.add_argument('--gen_len', type=int, help='Length of generation')
    parser.add_argument('--temp', type=float, help='Generation temperature')
    parser.add_argument('--topk', type=int, help='Top-k sampling')

    parser.set_defaults(
        gpu=True,
        num=2000,
        mem_len=512,
        gen_len=1024,
        temp=0.95,
        topk=32)

    args = parser.parse_args()

    model_fp = os.path.join(args.model_dir, 'model.pt')
    out_dir = os.path.join(args.model_dir, args.gen_dir)
    os.makedirs(os.path.join(out_dir, 'midi'), exist_ok=True)
    os.makedirs(os.path.join(out_dir, 'txt'), exist_ok=True)
    device = torch.device('cuda' if args.gpu else 'cpu')

    # Load the best saved model.
    with open(model_fp, 'rb') as f:
        model = torch.load(f)
    model.backward_compatible()
    model = model.to(device)

    # Make sure model uses vanilla softmax.
    if model.sample_softmax > 0:
        raise NotImplementedError()
    if model.crit.n_clusters != 0:
        raise NotImplementedError()

    # Load the vocab.
    vocab = Vocab(vocab_file='./data/vocab.txt', special=[])
    vocab.build_vocab()

    # Load the tokenizer
    tokenizer = MidiTokenizer()

    # Generate.
    controls = ['<animated_cartoon>', '<lovely>', '<hk_tw>', '<classic>',
                '<jazz>', '<usa_euro_pop>', '<christmas>', '<games>']
    if args.no_control:
        control_list = ['<cls>'] * args.num
    else:
        control_list = [random.choice(controls) for i in range(args.num)]

    i = 0
    while i < args.num:
        if i == int(int(args.num)/2):
            print('Change default arguments.')
            parser.set_defaults(
                mem_len=896,
                gen_len=2048,
                temp=1.2,
                topk=5)

        c = control_list[i]
        sampler = TokenSampler(model, device, mem_len=args.mem_len)
        seq = [vocab.sym2idx[c]]  # first token is the control code
        for _ in range(args.gen_len):
            token, _ = sampler.sample_next_token_updating_mem(
                last_token=seq[-1], temp=args.temp, topk=args.topk)
            seq.append(token)

        # Save midi
        sym_list = [vocab.idx2sym[t] for t in seq]
        midi_out_fp = os.path.join(out_dir, 'midi', str(i).zfill(4) + c[1:-1] + '.mid')
        if tokenizer.de_tokenize(sym_list, midi_out_fp):
            txt_out_fp = os.path.join(out_dir, 'txt', str(i).zfill(4) + c[1:-1] + '.txt')
            with open(txt_out_fp, 'w') as f:
                f.write('\n'.join(sym_list))
            i += 1
    print("Done.")
