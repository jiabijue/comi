import argparse
import itertools
import os
import time

import math
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from data_utils import get_lm_corpus
from transformer_xl import MemTransformerLM
from utils.data_parallel import BalancedDataParallel
from utils.exp_utils import create_exp_dir

# %% Arguments
parser = argparse.ArgumentParser()
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
parser.add_argument('--n_layer', type=int, default=12, help='number of total layers')
parser.add_argument('--d_model', type=int, default=512, help='model dimension')
parser.add_argument('--n_head', type=int, default=8, help='number of heads')
parser.add_argument('--d_head', type=int, default=64, help='head dimension')
parser.add_argument('--d_inner', type=int, default=2048, help='inner dimension in FF')
parser.add_argument('--dropout', type=float, default=0.1, help='global dropout rate')
parser.add_argument('--dropatt', type=float, default=0.0, help='attention probability dropout rate')
parser.add_argument('--d_embed', type=int, default=-1, help='embedding dimension')
parser.add_argument('--tied', type=bool, default=True, help='tie the word embedding and softmax weights')
parser.add_argument('--div_val', type=int, default=1, help='divident value for adaptive input and softmax')
parser.add_argument('--pre_lnorm', action='store_true', help='apply LayerNorm to the input instead of the output')

parser.add_argument('--tgt_len', type=int, default=512, help='number of tokens to predict')
parser.add_argument('--eval_tgt_len', type=int, default=128, help='number of tokens to predict for evaluation')
parser.add_argument('--ext_len', type=int, default=0, help='length of the extended context, must >= 0.')
parser.add_argument('--mem_len', type=int, default=512, help='length of the retained previous heads')
parser.add_argument('--same_length', action='store_true', help='use the same attn length for all tokens')
parser.add_argument('--attn_type', type=int, default=0,
                    help='attention type. 0 for ours, 1 for Shaw et al, 2 for Vaswani et al, 3 for Al Rfou et al.')
parser.add_argument('--clamp_len', type=int, default=-1, help='use the same pos embeddings after clamp_len')
parser.add_argument('--sample_softmax', type=int, default=-1, help='number of samples in sampled softmax')
parser.add_argument('--adaptive', action='store_true', help='use adaptive softmax')
##
parser.add_argument('--init', default='normal', type=str, help='parameter initializer to use.')
parser.add_argument('--emb_init', default='normal', type=str, help='parameter initializer to use.')
parser.add_argument('--init_range', type=float, default=0.1,
                    help='parameters initialized by U(-init_range, init_range)')
parser.add_argument('--emb_init_range', type=float, default=0.01,
                    help='parameters initialized by U(-init_range, init_range)')
parser.add_argument('--init_std', type=float, default=0.02, help='parameters initialized by N(0, init_std)')
parser.add_argument('--proj_init_std', type=float, default=0.01, help='parameters initialized by N(0, init_std)')

# Training args
parser.add_argument('--batch_size', type=int, default=12)
parser.add_argument('--eval_batch_size', type=int, default=8)
parser.add_argument('--max_step', type=int, default=120000, help='upper epoch limit')
parser.add_argument('--max_eval_steps', type=int, default=160, help='max eval steps')
parser.add_argument('--lr', type=float, default=0.0001, help='initial learning rate')
parser.add_argument('--clip', type=float, default=0.25, help='gradient clipping')
parser.add_argument('--log_interval', type=int, default=50, help='report interval')
parser.add_argument('--eval_interval', type=int, default=1000, help='evaluation interval')
parser.add_argument('--optim', default='adam', type=str, choices=['adam', 'sgd', 'adagrad'], help='optimizer to use.')
parser.add_argument('--mom', type=float, default=0.0, help='momentum for sgd')
parser.add_argument('--scheduler', default='cosine', type=str, choices=['cosine', 'inv_sqrt', 'dev_perf', 'constant'],
                    help='lr scheduler to use.')
parser.add_argument('--eta_min', type=float, default=0.0, help='min learning rate for cosine scheduler')
parser.add_argument('--patience', type=int, default=0, help='patience')
parser.add_argument('--warmup_step', type=int, default=0, help='upper epoch limit')
parser.add_argument('--decay_rate', type=float, default=0.5, help='decay factor when ReduceLROnPlateau is used')
parser.add_argument('--lr_min', type=float, default=0.0, help='minimum learning rate during annealing')
parser.add_argument('--clip_nonemb', action='store_true', help='only clip the gradient of non-embedding params')
parser.add_argument('--use_cuda', action='store_true')
parser.add_argument('--multi_gpu', action='store_true', help='use multiple GPU')
parser.add_argument('--gpu0_bsz', type=int, default=-1, help='batch size on gpu 0')
parser.add_argument('--batch_chunk', type=int, default=1, help='split batch into chunks to save memory')
# Experiment args
parser.add_argument('--work_dir', default='train', type=str, help='experiment directory.')
parser.add_argument('--debug', action='store_true', help='run in debug mode (do not create exp dir)')
parser.add_argument('--restart', action='store_true', help='restart training from the saved checkpoint')
parser.add_argument('--restart_dir', type=str, default='', help='restart dir')

parser.add_argument('--varlen', action='store_true',
                    help='use variable length')
args = parser.parse_args()

if args.d_embed < 0:
    args.d_embed = args.d_model

assert args.ext_len >= 0, 'extended context length must be non-negative'
assert args.batch_size % args.batch_chunk == 0

args.work_dir = os.path.join('exp', args.dataset, args.work_dir)
logging = create_exp_dir(args.work_dir,
                         scripts_to_save=['data_config.py', 'train.py', 'transformer_xl.py'],
                         debug=args.debug)

# Set the random seed manually for reproducibility.
np.random.seed(args.seed)
torch.manual_seed(args.seed)
if torch.cuda.is_available() and args.use_cuda:
    torch.cuda.manual_seed_all(args.seed)  # for multi-gpu

device = torch.device('cuda' if args.use_cuda else 'cpu')

# %% Load Data
args.data_corpus = os.path.join('./data', args.dataset, args.data_corpus)
corpus = get_lm_corpus(args.data_corpus, args.dataset)
args.n_token = len(corpus.vocab)

tr_iter = corpus.get_iterator('train', args.batch_size, args.tgt_len,
                              device=device, ext_len=args.ext_len,
                              augment_instrument=args.augment_instrument,
                              augment_pitch=args.augment_pitch,
                              augment_duration=args.augment_duration,
                              augment_velocity=args.augment_velocity)
va_iter = corpus.get_iterator('valid', args.eval_batch_size, args.eval_tgt_len,
                              device=device, ext_len=args.ext_len)
te_iter = corpus.get_iterator('test', args.eval_batch_size, args.eval_tgt_len,
                              device=device, ext_len=args.ext_len)

# adaptive softmax / embedding
cutoffs, tie_projs = [], [False]  # jbj:干嘛用的？


# %% Build Model
def init_weight(weight):
    if args.init == 'uniform':
        nn.init.uniform_(weight, -args.init_range, args.init_range)
    elif args.init == 'normal':
        nn.init.normal_(weight, 0.0, args.init_std)


def init_bias(bias):
    nn.init.constant_(bias, 0.0)


def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Linear') != -1:
        if hasattr(m, 'weight') and m.weight is not None:
            init_weight(m.weight)
        if hasattr(m, 'bias') and m.bias is not None:
            init_bias(m.bias)
    elif classname.find('AdaptiveEmbedding') != -1:
        if hasattr(m, 'emb_projs'):
            for i in range(len(m.emb_projs)):
                if m.emb_projs[i] is not None:
                    nn.init.normal_(m.emb_projs[i], 0.0, args.proj_init_std)
    elif classname.find('Embedding') != -1:
        if hasattr(m, 'weight'):
            init_weight(m.weight)
    elif classname.find('ProjectedAdaptiveLogSoftmax') != -1:
        if hasattr(m, 'cluster_weight') and m.cluster_weight is not None:
            init_weight(m.cluster_weight)
        if hasattr(m, 'cluster_bias') and m.cluster_bias is not None:
            init_bias(m.cluster_bias)
        if hasattr(m, 'out_projs'):
            for i in range(len(m.out_projs)):
                if m.out_projs[i] is not None:
                    nn.init.normal_(m.out_projs[i], 0.0, args.proj_init_std)
    elif classname.find('LayerNorm') != -1:
        if hasattr(m, 'weight'):
            nn.init.normal_(m.weight, 1.0, args.init_std)
        if hasattr(m, 'bias') and m.bias is not None:
            init_bias(m.bias)
    elif classname.find('TransformerLM') != -1:
        if hasattr(m, 'r_emb'):
            init_weight(m.r_emb)
        if hasattr(m, 'r_w_bias'):
            init_weight(m.r_w_bias)
        if hasattr(m, 'r_r_bias'):
            init_weight(m.r_r_bias)
        if hasattr(m, 'r_bias'):
            init_bias(m.r_bias)


def update_dropout(m):
    classname = m.__class__.__name__
    if classname.find('Dropout') != -1:
        if hasattr(m, 'p'):
            m.p = args.dropout


def update_dropatt(m):
    if hasattr(m, 'dropatt'):
        m.dropatt.p = args.dropatt


if args.restart:
    with open(os.path.join(args.restart_dir, 'model.pt'), 'rb') as f:
        model = torch.load(f)
        model = model.float()
    model.apply(update_dropout)
    model.apply(update_dropatt)
else:
    model = MemTransformerLM(args.n_token, args.n_layer, args.d_model, args.n_head,
                             args.d_head, args.d_inner, args.dropout, args.dropatt,
                             d_embed=args.d_embed, cutoffs=cutoffs, tie_projs=tie_projs,
                             tie_weight=args.tied, div_val=args.div_val,
                             pre_lnorm=args.pre_lnorm, tgt_len=args.tgt_len,
                             ext_len=args.ext_len, mem_len=args.mem_len,
                             same_length=args.same_length, attn_type=args.attn_type,
                             clamp_len=args.clamp_len, sample_softmax=args.sample_softmax).to(device)
    model.apply(weights_init)
    model.word_emb.apply(weights_init)  # ensure embedding init is not overridden by out_layer in case of weight sharing

args.n_all_param = sum([p.nelement() for p in model.parameters()])
args.n_nonemb_param = sum([p.nelement() for p in model.layers.parameters()])

if args.multi_gpu:
    model = model.to(device)
    if args.gpu0_bsz >= 0:
        para_model = BalancedDataParallel(args.gpu0_bsz // args.batch_chunk,
                                          model, dim=1).to(device)
    else:
        para_model = nn.DataParallel(model, dim=1).to(device)
else:
    para_model = model.to(device)

#### optimizer
if args.optim.lower() == 'sgd':
    if args.sample_softmax > 0:
        dense_params, sparse_params = [], []
        for param in model.parameters():
            if param.size() == model.word_emb.weight.size():
                sparse_params.append(param)
            else:
                dense_params.append(param)
        optimizer_sparse = optim.SGD(sparse_params, lr=args.lr * 2)
        optimizer = optim.SGD(dense_params, lr=args.lr, momentum=args.mom)
    else:
        optimizer = optim.SGD(model.parameters(), lr=args.lr,
                              momentum=args.mom)
elif args.optim.lower() == 'adam':
    if args.sample_softmax > 0:
        dense_params, sparse_params = [], []
        for param in model.parameters():
            if param.size() == model.word_emb.weight.size():
                sparse_params.append(param)
            else:
                dense_params.append(param)
        optimizer_sparse = optim.SparseAdam(sparse_params, lr=args.lr)
        optimizer = optim.Adam(dense_params, lr=args.lr)
    else:
        optimizer = optim.Adam(model.parameters(), lr=args.lr)
elif args.optim.lower() == 'adagrad':
    optimizer = optim.Adagrad(model.parameters(), lr=args.lr)

#### scheduler
if args.scheduler == 'cosine':
    # here we do not set eta_min to lr_min to be backward compatible
    # because in previous versions eta_min is default to 0
    # rather than the default value of lr_min 1e-6
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer,
                                                     args.max_step, eta_min=args.eta_min)  # should use eta_min arg
    if args.sample_softmax > 0:
        scheduler_sparse = optim.lr_scheduler.CosineAnnealingLR(optimizer_sparse,
                                                                args.max_step,
                                                                eta_min=args.eta_min)  # should use eta_min arg
elif args.scheduler == 'inv_sqrt':
    # originally used for Transformer (in Attention is all you need)
    def lr_lambda(step):
        # return a multiplier instead of a learning rate
        if step == 0 and args.warmup_step == 0:
            return 1.
        else:
            return 1. / (step ** 0.5) if step > args.warmup_step \
                else step / (args.warmup_step ** 1.5)


    scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_lambda)
elif args.scheduler == 'dev_perf':
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer,
                                                     factor=args.decay_rate, patience=args.patience, min_lr=args.lr_min)
    if args.sample_softmax > 0:
        scheduler_sparse = optim.lr_scheduler.ReduceLROnPlateau(optimizer_sparse,
                                                                factor=args.decay_rate, patience=args.patience,
                                                                min_lr=args.lr_min)
elif args.scheduler == 'constant':
    pass

if args.restart:
    if os.path.exists(os.path.join(args.restart_dir, 'optimizer.pt')):
        with open(os.path.join(args.restart_dir, 'optimizer.pt'), 'rb') as f:
            opt_state_dict = torch.load(f)
            optimizer.load_state_dict(opt_state_dict)
    else:
        print('Optimizer was not saved. Start from scratch.')

logging('=' * 100)
for k, v in args.__dict__.items():
    logging('    - {} : {}'.format(k, v))
logging('=' * 100)
logging('#params = {}'.format(args.n_all_param))
logging('#non emb params = {}'.format(args.n_nonemb_param))


# %% Training
def evaluate(eval_iter, is_test=False):
    # Turn on evaluation mode which disables dropout.
    model.eval()

    # If the model does not use memory at all, make the ext_len longer.
    # Otherwise, make the mem_len longer and keep the ext_len the same.
    if args.mem_len == 0:
        model.reset_length(args.eval_tgt_len,
                           args.ext_len + args.tgt_len - args.eval_tgt_len, args.mem_len)
    else:
        model.reset_length(args.eval_tgt_len,
                           args.ext_len, args.mem_len + args.tgt_len - args.eval_tgt_len)

    # Evaluation
    total_len, total_loss = 0, 0.
    total_rand_loss = 0.
    with torch.no_grad():
        mems = tuple()
        for i, (data, target, seq_len) in enumerate(eval_iter):
            if 0 < args.max_eval_steps <= i:
                break
            ret = model(data, target, *mems)
            loss, mems = ret[0], ret[1:]
            loss = loss.mean()
            total_loss += seq_len * loss.float().item()
            if is_test:
                total_rand_loss += seq_len * model.compute_random_loss(data, target)
            total_len += seq_len

    # Switch back to the training mode
    model.reset_length(args.tgt_len, args.ext_len, args.mem_len)
    model.train()

    if is_test:
        return total_loss / total_len, total_rand_loss / total_len
    else:
        return total_loss / total_len


def train():
    # Turn on training mode which enables dropout.
    global train_step, train_loss, best_val_loss, eval_start_time, log_start_time
    model.train()
    if args.batch_chunk > 1:
        mems = [tuple() for _ in range(args.batch_chunk)]
    else:
        mems = tuple()
    train_iter = tr_iter.get_varlen_iter() if args.varlen else tr_iter
    for batch, (data, target, seq_len) in enumerate(train_iter):
        model.zero_grad()
        if args.batch_chunk > 1:
            data_chunks = torch.chunk(data, args.batch_chunk, 1)
            target_chunks = torch.chunk(target, args.batch_chunk, 1)
            for i in range(args.batch_chunk):
                data_i = data_chunks[i].contiguous()
                target_i = target_chunks[i].contiguous()
                ret = para_model(data_i, target_i, *mems[i])
                loss, mems[i] = ret[0], ret[1:]
                loss = loss.float().mean().type_as(loss) / args.batch_chunk
                loss.backward()
                train_loss += loss.float().item()
        else:
            ret = para_model(data, target, *mems)
            loss, mems = ret[0], ret[1:]
            loss = loss.float().mean().type_as(loss)
            loss.backward()
            train_loss += loss.float().item()

        torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip)

        optimizer.step()
        if args.sample_softmax > 0:
            optimizer_sparse.step()

        # step-wise learning rate annealing
        train_step += 1
        if args.scheduler in ['cosine', 'constant', 'dev_perf']:
            # linear warmup stage
            if train_step < args.warmup_step:
                curr_lr = args.lr * train_step / args.warmup_step
                optimizer.param_groups[0]['lr'] = curr_lr
                if args.sample_softmax > 0:
                    optimizer_sparse.param_groups[0]['lr'] = curr_lr * 2
            else:
                if args.scheduler == 'cosine':
                    scheduler.step()
                    if args.sample_softmax > 0:
                        scheduler_sparse.step(train_step)
        elif args.scheduler == 'inv_sqrt':
            scheduler.step(train_step)

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
            val_loss = evaluate(va_iter)
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

            # dev-performance based learning rate annealing
            if args.scheduler == 'dev_perf':
                scheduler.step(val_loss)
                if args.sample_softmax > 0:
                    scheduler_sparse.step(val_loss)

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
test_loss, test_rand_loss = evaluate(te_iter, True)
logging('=' * 100)
logging('| End of training | random test loss {:4.4f} | random test ppl {:7.4f} | test loss {:4.4f} | test ppl {:7.4f}'
        .format(test_rand_loss, math.exp(test_rand_loss), test_loss, math.exp(test_loss)))
logging('=' * 100)
