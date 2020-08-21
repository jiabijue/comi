import argparse
import itertools
import os
import random
import time

import math
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from data_utils import get_lm_corpus
from utils.exp_utils import create_exp_dir, get_logger

# os.environ['CUDA_VISIBLE_DEVICES'] = '1'


# %% Arguments
parser = argparse.ArgumentParser()
parser.add_argument('--mode', type=str, default='train', choices=['train', 'eval', 'gen'])

# Data args
parser.add_argument('--dataset', type=str, default='midicn', choices=['lakh', 'midicn', 'mirex-like'])
parser.add_argument('--data_corpus', type=str, default='txt_3', help='location of the data corpus')
parser.add_argument('--augment_instrument', action='store_true',
                    help='augment instrument by randomly choosing an instrument in the same instrument family')
parser.add_argument('--augment_pitch', action='store_true', help='transpose pitch randomly by -6 to +6 semitones')
parser.add_argument('--augment_duration', action='store_true', help='stretch duration randomly by 95% to 105%')
parser.add_argument('--augment_velocity', action='store_true', help='augment velocity randomly by -20 to +20 semitones')
parser.add_argument('--seed', type=int, default=7777, help='random seed')

# Model args
parser.add_argument('--unit_type', default='lstm')
parser.add_argument('--d_embed', type=int, default=512, help='embedding dimension')
parser.add_argument('--hidden_size', type=int, default=1024, help='RNN hidden size')
parser.add_argument('--num_layers', type=int, default=6, help='RNN layers')
parser.add_argument('--dropout', type=float, default=0.75, help='RNN dropout value')
parser.add_argument('--bidirectional', type=bool, default=False, help='use bidirectional RNN')

parser.add_argument('--tgt_len', type=int, default=512, help='number of tokens to predict')
parser.add_argument('--eval_tgt_len', type=int, default=128, help='number of tokens to predict for evaluation')
parser.add_argument('--ext_len', type=int, default=0, help='length of the extended context, must >= 0.')

# Training args
parser.add_argument('--batch_size', type=int, default=64)
parser.add_argument('--eval_batch_size', type=int, default=16)
parser.add_argument('--max_step', type=int, default=40000, help='upper epoch limit')
parser.add_argument('--max_eval_steps', type=int, default=160, help='max eval steps')
parser.add_argument('--lr', type=float, default=0.0001)
parser.add_argument('--clip', type=float, default=0.25, help='gradient clipping')
parser.add_argument('--log_interval', type=int, default=50, help='report interval')
parser.add_argument('--eval_interval', type=int, default=1000, help='evaluation interval')
parser.add_argument('--eta_min', type=float, default=0.0, help='min learning rate for cosine scheduler')
parser.add_argument('--use_cuda', action='store_true')

# Generation args
parser.add_argument('--num', type=int, default=1000, help='number of samples to generate.')
parser.add_argument('--gen_dir', type=str, default='gen', help='Directory to put generated samples')
parser.add_argument('--gen_len', type=int, default=1024, help='Length of generation')

# Experiment args
parser.add_argument('--work_dir', default='train', type=str, help='experiment directory.')
parser.add_argument('--debug', action='store_true', help='run in debug mode (do not create exp dir)')
parser.add_argument('--restart', action='store_true', help='restart training from the saved checkpoint')
parser.add_argument('--restart_dir', type=str, default='', help='restart dir')

args = parser.parse_args()

args.work_dir = os.path.join('exp', args.dataset, args.unit_type, args.work_dir)

if args.mode == 'train':
    logging = create_exp_dir(args.work_dir,
                             scripts_to_save=['data_config.py', 'main_rnn.py'],
                             debug=args.debug)
elif args.mode == 'eval':
    # Get logger
    logging = get_logger(os.path.join(args.model_dir, 'log.txt'), log_=not args.debug)
elif args.mode == 'gen':
    out_dir = os.path.join(args.work_dir, args.gen_dir)
    os.makedirs(os.path.join(out_dir, 'midi'), exist_ok=True)
    os.makedirs(os.path.join(out_dir, 'txt'), exist_ok=True)
    # Load the tokenizer
    from tokenization_by_time import MidiTokenizer

    tokenizer = MidiTokenizer()

device = torch.device('cuda' if args.use_cuda else 'cpu')

if args.mode == 'train':
    # Set the random seed manually for reproducibility.
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available() and args.use_cuda:
        torch.cuda.manual_seed_all(args.seed)  # for multi-gpu

# %% Load Data
if args.mode != 'gen':
    args.data_corpus = os.path.join('./data', args.dataset, args.data_corpus)
    corpus = get_lm_corpus(args.data_corpus, args.dataset)
    vocab = corpus.vocab
else:
    from utils.vocabulary import Vocab
    vocab = Vocab(vocab_file='./data/vocab.txt', special=[])
    vocab.build_vocab()
args.n_token = len(vocab)

if args.mode == 'train':
    tr_iter = corpus.get_iterator('train', args.batch_size, args.tgt_len,
                                  device=device, ext_len=args.ext_len,
                                  augment_instrument=args.augment_instrument,
                                  augment_pitch=args.augment_pitch,
                                  augment_duration=args.augment_duration,
                                  augment_velocity=args.augment_velocity)
if args.mode != 'gen':
    va_iter = corpus.get_iterator('valid', args.eval_batch_size, args.eval_tgt_len,
                                  device=device, ext_len=args.ext_len)
    te_iter = corpus.get_iterator('test', args.eval_batch_size, args.eval_tgt_len,
                                  device=device, ext_len=args.ext_len)

# adaptive softmax / embedding
cutoffs, tie_projs = [], [False]  # jbj:干嘛用的？

# %% Define Model
d_proj = args.d_embed


class RNN(nn.Module):
    def __init__(self, n_token, d_embed, d_proj, cutoffs,
                 hidden_size, num_layers, dropout, bidirectional=True, unit_type='lstm'):
        super(RNN, self).__init__()

        self.drop = nn.Dropout(dropout)

        # self.word_emb = AdaptiveEmbedding(n_token, d_embed, d_proj, cutoffs)
        self.word_emb = nn.Embedding(n_token, d_embed)

        num_directions = 2 if bidirectional else 1

        if unit_type == 'lstm':
            self.rnn_layers = nn.LSTM(d_embed, hidden_size, num_layers, dropout=dropout, bidirectional=bidirectional)
        elif unit_type == 'gru':
            self.rnn_layers = nn.GRU(d_embed, hidden_size, num_layers, dropout=dropout, bidirectional=bidirectional)

        self.out_layer = nn.Linear(num_directions * hidden_size, n_token)

    def forward(self, data):
        word_emb = self.drop(self.word_emb(data))
        rnn_out, _ = self.rnn_layers(word_emb)
        rnn_out = self.drop(rnn_out)
        output = self.out_layer(rnn_out)  # (seq_len, bsz, n_tokens)
        return output


if args.mode == 'train':
    if args.restart:
        with open(os.path.join(args.restart_dir, 'model.pt'), 'rb') as f:
            model = torch.load(f)
    else:
        model = RNN(args.n_token, args.d_embed, d_proj, cutoffs,
                    args.hidden_size, args.num_layers, args.dropout, args.bidirectional, args.unit_type)
else:
    # Load the best saved model.
    with open(os.path.join(args.work_dir, 'model.pt'), 'rb') as f:
        model = torch.load(f)
model = model.to(device)

if args.mode == 'train':
    args.n_all_param = sum([p.nelement() for p in model.parameters()])

    logging('=' * 100)
    for k, v in args.__dict__.items():
        logging('    - {} : {}'.format(k, v))
    logging('=' * 100)
    logging('#params = {}'.format(args.n_all_param))

# %% Training
if args.mode == 'train':
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, args.max_step, eta_min=args.eta_min)


    def validate(eval_iter):
        # Turn on evaluation mode which disables dropout.
        model.eval()

        # Evaluation
        total_len, total_loss = 0, 0.
        with torch.no_grad():
            for i, (data, target, seq_len) in enumerate(eval_iter):
                if 0 < args.max_eval_steps <= i:
                    break
                output = model(data)
                output_flat = output.view(-1, args.n_token)
                loss = criterion(output_flat, target.view(-1))
                total_loss += seq_len * loss.item()
                total_len += seq_len

        # Switch back to the training mode
        model.train()

        return total_loss / total_len


    def train():
        # Turn on training mode which enables dropout.
        global train_step, train_loss, best_val_loss, eval_start_time, log_start_time
        model.train()

        for batch, (data, target, seq_len) in enumerate(tr_iter):
            model.zero_grad()

            output = model(data)
            output_flat = output.view(-1, args.n_token)
            loss = criterion(output_flat, target.view(-1))
            loss.backward()

            train_loss += loss.item()

            torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip)

            optimizer.step()

            # step-wise learning rate annealing
            train_step += 1
            scheduler.step()

            if train_step % args.log_interval == 0:
                cur_loss = train_loss / args.log_interval
                elapsed = time.time() - log_start_time
                log_str = '| epoch {:3d} step {:>4d} | {:>3d} batches | lr {:.3g} | ms/batch {:5.2f} | loss {:4.4f}' \
                    .format(epoch, train_step, batch + 1, optimizer.param_groups[0]['lr'],
                            elapsed * 1000 / args.log_interval, cur_loss)
                log_str += ' | ppl {:7.4f}'.format(math.exp(cur_loss))
                logging(log_str)
                train_loss = 0
                log_start_time = time.time()

            if train_step == 1 or train_step % args.eval_interval == 0:
                val_loss = validate(va_iter)
                logging('-' * 100)
                log_str = '| Eval {:3d} at step {:>4d} | time: {:3.2f}s | valid loss {:4.4f}' \
                    .format(train_step // args.eval_interval, train_step, (time.time() - eval_start_time), val_loss)
                log_str += ' | valid ppl {:7.4f}'.format(math.exp(val_loss))
                logging(log_str)
                logging('-' * 100)
                # Save the model if the validation loss is the best we've seen so far.
                if not best_val_loss or val_loss < best_val_loss:
                    if not args.debug:
                        with open(os.path.join(args.work_dir, 'model.pt'), 'wb') as f:
                            torch.save(model, f)
                        with open(os.path.join(args.work_dir, 'optimizer.pt'), 'wb') as f:
                            torch.save(optimizer.state_dict(), f)
                    best_val_loss = val_loss

                eval_start_time = time.time()

            if train_step == args.max_step:
                break


    # Loop over epochs.
    train_step = 0
    train_loss = 0
    best_val_loss = None

    log_start_time = time.time()
    eval_start_time = time.time()

    # At any point you can hit Ctrl + C to break out of training early.
    try:
        for epoch in itertools.count(start=1):
            train()
            if train_step == args.max_step:
                logging('-' * 100)
                logging('End of training')
                break
    except KeyboardInterrupt:
        logging('-' * 100)
        logging('Exiting from training early')

    # Load the best saved model.
    with open(os.path.join(args.work_dir, 'model.pt'), 'rb') as f:
        model = torch.load(f)

    # Run on test data.
    test_loss = validate(te_iter)
    logging('=' * 100)
    logging('| End of training | test loss {:4.4f} | test ppl {:7.4f}'.format(
        test_loss, math.exp(test_loss)))
    logging('=' * 100)

# %% Evaluation only
elif args.mode == 'eval':
    def repackage_hidden(h):
        """Wraps hidden states in new Tensors, to detach them from their history."""

        if isinstance(h, torch.Tensor):
            return h.detach()
        else:
            return tuple(repackage_hidden(v) for v in h)

    def evaluate(eval_iter):
        # Turn on evaluation mode which disables dropout.
        model.eval()

        total_len, total_loss = 0, 0.
        start_time = time.time()
        with torch.no_grad():
            for idx, (data, target, seq_len) in enumerate(eval_iter):
                output = model(data)
                # hidden = repackage_hidden(hidden)
                output_flat = output.view(-1, args.n_token)
                loss = criterion(output_flat, target.view(-1))
                total_loss += seq_len * loss.item()
                total_len += seq_len
            total_time = time.time() - start_time
        logging('Time : {:.2f}s, {:.2f}ms/segment'.format(
            total_time, 1000 * total_time / (idx + 1)))
        return total_loss / total_len


    # Run on test data.
    if args.split == 'all':
        test_loss = evaluate(te_iter)
        valid_loss = evaluate(va_iter)
    elif args.split == 'valid':
        valid_loss = evaluate(va_iter)
        test_loss = None
    elif args.split == 'test':
        test_loss = evaluate(te_iter)
        valid_loss = None


    def format_log(loss, split):
        log_str = '| {0} loss {1:4.4f} | {0} ppl {2:9.4f} '.format(
            split, loss, math.exp(loss))
        return log_str


    log_str = ''
    if valid_loss is not None:
        log_str += format_log(valid_loss, 'valid')
    if test_loss is not None:
        log_str += format_log(test_loss, 'test')

    logging('=' * 100)
    logging(log_str)
    logging('=' * 100)

# %%  Generation
elif args.mode == 'gen':
    # Generate.
    controls = ['<animated_cartoon>', '<lovely>', '<hk_tw>', '<classic>',
                '<jazz>', '<usa_euro_pop>', '<christmas>', '<games>']
    control_list = [random.choice(controls) for i in range(args.num)]
    temperature = 0.95

    with torch.no_grad():
        i = 0
        while i < args.num:
            if i == int(int(args.num) / 2):
                print('Change default arguments.')
                temperature = 1.2
                args.gen_len = 2048
            c = control_list[i]
            input_ = torch.tensor(vocab.sym2idx[c], dtype=torch.long).reshape(1, 1).to(device)
            seq = [c]
            for _ in range(args.gen_len):
                output = model(input_)
                token_weights = output.squeeze().div(temperature).exp().cpu()
                token_idx = torch.multinomial(token_weights, 1)[0]
                input_.fill_(token_idx)

                token = vocab.idx2sym[token_idx]
                seq.append(token)

            # Save midi
            midi_out_fp = os.path.join(out_dir, 'midi', str(i).zfill(3) + c[1:-1] + '.mid')
            if tokenizer.de_tokenize(seq, midi_out_fp):
                txt_out_fp = os.path.join(out_dir, 'txt', str(i).zfill(3) + c[1:-1] + '.txt')
                with open(txt_out_fp, 'w') as f:
                    f.write('\n'.join(seq))
            i += 1
    print("Done.")
