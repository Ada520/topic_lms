import argparse
#import ipdb
import time
import math
import numpy as np
import torch
import torch.nn as nn
from torch.autograd import Variable
import pickle
import data
import model
import os
from gensim import models
import gensim
import itertools

#os.environ["CUDA_VISIBLE_DEVICES"] = "1,2,3"
"""
run using python main.py --batch_size 20 --data data/penn --dropouti 0.4 --dropouth 0.3 --seed 141 --epoch 100 --save apnews.pt
"""

from utils import batchify, get_batch, repackage_hidden

parser = argparse.ArgumentParser(description='PyTorch PennTreeBank RNN/LSTM Language Model')
parser.add_argument('--data', type=str, default='data/penn/',
                    help='location of the data corpus')
parser.add_argument('--model', type=str, default='LSTM',
                    help='type of recurrent net (LSTM, QRNN, GRU)')
parser.add_argument('--emsize', type=int, default=400,
                    help='size of word embeddings')
parser.add_argument('--nhid', type=int, default=1150,
                    help='number of hidden units per layer')
parser.add_argument('--nlayers', type=int, default=3,
                    help='number of layers')
parser.add_argument('--lr', type=float, default=30,
                    help='initial learning rate')
parser.add_argument('--clip', type=float, default=0.25,
                    help='gradient clipping')
parser.add_argument('--epochs', type=int, default=8000,
                    help='upper epoch limit')
parser.add_argument('--batch_size', type=int, default=20, metavar='N',
                    help='batch size')
parser.add_argument('--bptt', type=int, default=70,
                    help='sequence length')
parser.add_argument('--dropout', type=float, default=0.4,
                    help='dropout applied to layers (0 = no dropout)')
parser.add_argument('--dropouth', type=float, default=0.3,
                    help='dropout for rnn layers (0 = no dropout)')
parser.add_argument('--dropouti', type=float, default=0.65,
                    help='dropout for input embedding layers (0 = no dropout)')
parser.add_argument('--dropoute', type=float, default=0.1,
                    help='dropout to remove words from embedding layer (0 = no dropout)')
parser.add_argument('--wdrop', type=float, default=0.5,
                    help='amount of weight dropout to apply to the RNN hidden to hidden matrix')
parser.add_argument('--tied', action='store_false',
                    help='tie the word embedding and softmax weights')
parser.add_argument('--seed', type=int, default=1111,
                    help='random seed')
parser.add_argument('--nonmono', type=int, default=5,
                    help='random seed')
parser.add_argument('--cuda', action='store_false', default=True,
                    help='use CUDA')
parser.add_argument('--log-interval', type=int, default=200, metavar='N',
                    help='report interval')
randomhash = ''.join(str(time.time()).split('.'))
parser.add_argument('--save', type=str,  default=randomhash+'.pt',
                    help='path to save the final model')
parser.add_argument('--alpha', type=float, default=2,
                    help='alpha L2 regularization on RNN activation (alpha = 0 means no regularization)')
parser.add_argument('--beta', type=float, default=1,
                    help='beta slowness regularization applied on RNN activiation (beta = 0 means no regularization)')
parser.add_argument('--wdecay', type=float, default=1.2e-6,
                    help='weight decay applied to all weights')
parser.add_argument('--mit-topic', type=bool, default=True,
                    help='with additional topic embedding')
parser.add_argument('--domain', type=str, default='apnews',
                    help='with additional topic embedding')
args = parser.parse_args()

# Set the random seed manually for reproducibility.
np.random.seed(args.seed)
torch.manual_seed(args.seed)
if torch.cuda.is_available():
    if not args.cuda:
        print("WARNING: You have a CUDA device, so you should probably run with --cuda")
    else:
        torch.cuda.manual_seed(args.seed)

###############################################################################
# Load data
###############################################################################
print ('Processing domain: '+ args.domain)

seq_len = 30
train_path = os.path.expanduser('~/topic_lms/data/' + args.domain + '/train_transform.pkl')
valid_path = os.path.expanduser('~/topic_lms/data/'+ args.domain + '/val_transform.pkl')
test_path = os.path.expanduser('~/topic_lms/data/' + args.domain + '/test_transform.pkl')
vocab = os.path.expanduser('~/topic_lms/data/' + args.domain + '/vocab.pkl')
lda_path = os.path.expanduser('~/topic_lms/data/' + args.domain + '/lda_models/lda_model')
#path to gensim dictionary used to create lda model
lda_dict_path = os.path.expanduser('~/topic_lms/data/' + args.domain + '/lda_models/lda_dict')
#fast_text_file = os.path.expanduser('~/topic_lms/data/imdb/apnews_ft.vec')
eval_batch_size = 64
test_batch_size = 64

with open(train_path, 'rb') as f:
    train_data = pickle.load(f)

with open(valid_path, 'rb') as f:
    valid_data = pickle.load(f)

with open(test_path, 'rb') as f:
    test_data = pickle.load(f)

with open(vocab, 'rb') as f:
    vocab = pickle.load(f)

w2id = vocab
idx2word = {v: k for k, v in w2id.items()}
ntokens = len(vocab) + 1
lda_model = models.LdaModel.load(lda_path)
#load the lda dictionary
lda_dictionary = gensim.corpora.Dictionary.load(lda_dict_path)

#print (valid_data.shape, train_data.shape, test_data.shape)
#train_data = batchify(corpus.train, args.batch_size, args)

#val_data = batchify(corpus.valid, eval_batch_size, args)
#test_data = batchify(corpus.test, test_batch_size, args)

###############################################################################
# Build the model
###############################################################################


if args.mit_topic:
    model = model.RNNModel_mit_topic(args.model, ntokens, args.emsize, args.nhid, args.nlayers, args.dropout, args.dropouth, args.dropouti, args.dropoute, args.wdrop, args.tied, topic_size=50)
else:
    model = model.RNNModel(args.model, ntokens, args.emsize, args.nhid, args.nlayers, args.dropout, args.dropouth, args.dropouti, args.dropoute, args.wdrop, args.tied)
if args.cuda:
    model.cuda()
total_params = sum(x.size()[0] * x.size()[1] if len(x.size()) > 1 else x.size()[0] for x in model.parameters())
print(f'Args: {args}')
print(f'Model total parameters: {total_params}')

criterion = nn.CrossEntropyLoss(ignore_index=-1)

###############################################################################
# get lda vectors
###############################################################################


def get_lda_vec(lda_dict):
    """
    get lda vector
    :param lda_dict:
    :return:
    """
    lda_vec = np.zeros(50, dtype='float32')
    for id, val in lda_dict:
        lda_vec[id] = val
    return lda_vec


def get_id2word(idx, idx2w_dict):
    """
    get id2word mappings
    :param idx:
    :param idx2w_dict:
    :return:
    """
    try:
        return idx2w_dict[idx]
    except KeyError:
        return '__UNK__'


def get_theta(texts, lda, dictionari, idx2word):
    """
    get doc-topic distribution vector for all reviews
    :param texts:
    :param lda:e
    :param dictionari:
    :param idx2word:
    :return:
    """
    texts = np.transpose(texts)
    texts = [[get_id2word(idx, idx2word) for idx in sent] for sent in texts]
    review_alphas = np.array([get_lda_vec(lda[dictionari.doc2bow(sentence)]) for sentence in texts])
    return torch.from_numpy(review_alphas)


###############################################################################
# Training code
###############################################################################


def evaluate(data_source, batch_size=10):
    print("EVALUATION")
    # Turn on evaluation mode which disables dropout.
    model.eval()
    if args.model == 'QRNN':
        model.reset()
    total_loss = 0
    hidden = model.init_hidden(args.batch_size)
    #_, batch_len = data_source.shape
    #n_batches = (batch_len -1) // seq_len

    for batch_n in range(0, len(data_source)-args.batch_size, args.batch_size):
        sub = train_data[batch_n: batch_n + args.batch_size]
        padded = np.array(list(itertools.zip_longest(*sub, fillvalue=0))).T
        targets = np.roll(padded, -1)
        targets[:, -1] = 0
        if args.cuda:
            #data = Variable(torch.from_numpy(data_source[:, batch_n * seq_len: (batch_n + 1) * seq_len])).transpose(0, 1).cuda()
            #targets = Variable(torch.from_numpy(data_source[:, batch_n * seq_len + 1: (batch_n + 1) * seq_len + 1].transpose(1, 0).flatten())).cuda()
            data = Variable(torch.from_numpy(padded.T)).cuda()
            targets = Variable(torch.from_numpy(targets.T.flatten())).cuda()
        else:
            #data = Variable(torch.from_numpy(data_source[:, batch_n * seq_len: (batch_n + 1) * seq_len])).transpose(0, 1)
            #targets = Variable(torch.from_numpy(data_source[:, batch_n * seq_len + 1: (batch_n + 1) * seq_len + 1].transpose(1, 0).flatten()))
            data = Variable(torch.from_numpy(padded))
            targets = Variable(torch.from_numpy(targets.flatten()))
        #print len(data), len(targets)
        #print data.size()

        #print "evaluating!"
        #comment out this line to get the original lda vector
        if args.cuda:
            inp_topic = get_theta(data.data.cpu().numpy(), lda_model, lda_dictionary, idx2word).cuda()
            inp_topic = inp_topic.type(torch.cuda.FloatTensor)
        else:
            inp_topic = get_theta(data.data.cpu().numpy(), lda_model, lda_dictionary, idx2word)
            inp_topic = inp_topic.type(torch.FloatTensor)
        #inp_topic = torch.from_numpy(np.zeros((args.batch_size, 50))).cuda()

        topic_var = torch.autograd.Variable(inp_topic, requires_grad=False)

        # Starting each batch, we detach the hidden state from how it was previously produced.
        # If we didn't, the model would try backpropagating all the way to start of the dataset.
        hidden = repackage_hidden(hidden)
        optimizer.zero_grad()
        if args.mit_topic:
            output = model(data, topic_var, hidden)
        else:
            output = model(data, hidden)
        output_flat = output.view(-1, ntokens)
        total_loss += len(data) * criterion(output_flat, targets).data
        #hidden = repackage_hidden(hidden)
    return total_loss[0] / (data_source.shape[1])


def train():
    # Turn on training mode which enables dropout.
    if args.model == 'QRNN': model.reset()
    total_loss = 0
    start_time = time.time()
    hidden = model.init_hidden(args.batch_size)
    #m, batch_len = train_data.shape
    #n_batches = (batch_len -1) // seq_len
    data_len = len(train_data)
    for batch_n in range(0, data_len-args.batch_size, args.batch_size):
        bptt = args.bptt if np.random.random() < 0.95 else args.bptt / 2.

        lr2 = optimizer.param_groups[0]['lr']
        optimizer.param_groups[0]['lr'] = lr2 * seq_len / args.bptt
        sub = train_data[batch_n: batch_n + args.batch_size]
        seqlen = [len(dat) for dat in sub]
        padded = np.array(list(itertools.zip_longest(*sub, fillvalue=0))).T
        #print (padded.shape)
        targets = np.roll(padded, -1)
        targets[:, -1] = 0
        target_lens = [targets[i][:(seqlen[i])] for i in range(len(sub))]
        model.train()
        if args.cuda:
            # data = Variable(torch.from_numpy(train_data[:, batch_n * seq_len: (batch_n + 1) * seq_len])).transpose(0, 1).cuda()
            # targets = Variable(torch.from_numpy(train_data[:, batch_n * seq_len + 1: (batch_n + 1) * seq_len + 1].transpose(1, 0).flatten())).cuda()
            data = Variable(torch.from_numpy(padded.T)).cuda()
            targets = Variable(torch.from_numpy(targets.T.flatten())).cuda()
            target_lens = Variable(torch.from_numpy(np.concatenate(target_lens).ravel())).cuda()
        else:
            # data = Variable(torch.from_numpy(train_data[:, batch_n * seq_len: (batch_n + 1) * seq_len])).transpose(0, 1)
            # targets = Variable(torch.from_numpy(train_data[:, batch_n * seq_len + 1: (batch_n + 1) * seq_len + 1].transpose(1, 0).flatten()))
            data = Variable(torch.from_numpy(padded.T))
            targets = Variable(torch.from_numpy(targets.T.flatten()))
        #targets = targets.view(targets.numel())
        #data, targets = get_batch(train_data, i, args, seq_len=seq_len)
        #print ('next batch')
        #Comment out this line to get the original lda vector
        if args.cuda:
            inp_topic = get_theta(data.data.cpu().numpy(), lda_model, lda_dictionary, idx2word).cuda()
            inp_topic = inp_topic.type(torch.cuda.FloatTensor)
        else:
            inp_topic = get_theta(data.data.cpu().numpy(), lda_model, lda_dictionary, idx2word)
            inp_topic = inp_topic.type(torch.FloatTensor)
        #coment the vector with zeros
        #inp_topic = torch.from_numpy(np.zeros((args.batch_size, 50))).cuda()

        topic_var = torch.autograd.Variable(inp_topic, requires_grad=False)

        # Starting each batch, we detach the hidden state from how it was previously produced.
        # If we didn't, the model would try backpropagating all the way to start of the dataset.
        hidden = repackage_hidden(hidden)
        #print hidden
        optimizer.zero_grad()
        if args.mit_topic:
            output, rnn_hs, dropped_rnn_hs = model(data, topic_var, hidden, return_h=True)
        else:
            output, rnn_hs, dropped_rnn_hs = model(data, hidden, return_h=True)
        #print(output.size(), targets.size())
        #targets = np.array([np.array(sub[i][:(seqlen[i])], dtype=np.float32) for i in range(len(sub))])
        #print (output.view(-1, ntokens))
        #output = output.transpose(0, 1)
        output = output.cpu().data.numpy()
        output = np.transpose(output, (1, 0, 2))
        output = [output[:seqlen[i], :] for i in range(len(sub))]
        print (np.concatenate(output).ravel().shape)
        output = output.transpose(0, 1)
        raw_loss = criterion(output.view(-1, ntokens), targets)

        loss = raw_loss
        # Activiation Regularization
        loss = loss + sum(args.alpha * dropped_rnn_h.pow(2).mean() for dropped_rnn_h in dropped_rnn_hs[-1:])
        # Temporal Activation Regularization (slowness)
        loss = loss + sum(args.beta * (rnn_h[1:] - rnn_h[:-1]).pow(2).mean() for rnn_h in rnn_hs[-1:])
        #print (data.size())
        loss.backward()

        # `clip_grad_norm` helps prevent the exploding gradient problem in RNNs / LSTMs.
        torch.nn.utils.clip_grad_norm(model.parameters(), args.clip)
        optimizer.step()

        total_loss += raw_loss.data
        optimizer.param_groups[0]['lr'] = lr2
        if batch_n % args.log_interval == 0 and batch_n > 0:
            cur_loss = total_loss[0] / args.log_interval
            elapsed = time.time() - start_time
            print('| epoch {:3d} | {:5d}/{:5d} batches | lr {:02.2f} | ms/batch {:5.2f} | '
                    'loss {:5.2f} | ppl {:8.2f}'.format(
                epoch, batch_n, len(train_data) // args.batch_size, optimizer.param_groups[0]['lr'],
                elapsed * 1000 / args.log_interval, cur_loss, math.exp(cur_loss)))
            total_loss = 0
            start_time = time.time()
        ###
        #batch += 1
        #i += seq_len

# Loop over epochs.
lr = args.lr
best_val_loss = []
stored_loss = 100000000

# At any point you can hit Ctrl + C to break out of training early.
try:
    optimizer = torch.optim.SGD(model.parameters(), lr=args.lr, weight_decay=args.wdecay)
    for epoch in range(1, args.epochs+1):
        epoch_start_time = time.time()
        train()
        if 't0' in optimizer.param_groups[0]:
            tmp = {}
            for prm in model.parameters():
                tmp[prm] = prm.data.clone()
                prm.data = optimizer.state[prm]['ax'].clone()

            val_loss2 = evaluate(valid_data)
            print('-' * 89)
            print('| end of epoch {:3d} | time: {:5.2f}s | valid loss {:5.2f} | '
                    'valid ppl {:8.2f}'.format(epoch, (time.time() - epoch_start_time),
                                               val_loss2, math.exp(val_loss2)))
            print('-' * 89)

            if val_loss2 < stored_loss:
                #with open(args.save, 'wb') as f:
                torch.save(model, args.save)
                print('Saving Averaged!')
                stored_loss = val_loss2

            for prm in model.parameters():
                prm.data = tmp[prm].clone()

        else:
            val_loss = evaluate(valid_data, eval_batch_size)
            #print val_loss
            print('-' * 89)
            print('| end of epoch {:3d} | time: {:5.2f}s | valid loss {:5.2f} | '
                    'valid ppl {:8.2f}'.format(epoch, (time.time() - epoch_start_time),
                                               val_loss, math.exp(val_loss)))
            print('-' * 89)

            if val_loss < stored_loss:
                #with open(args.save, 'wb') as f:
                torch.save(model.state_dict(), args.save)
                print('Saving Normal!')
                stored_loss = val_loss

            if 't0' not in optimizer.param_groups[0] and (len(best_val_loss)>args.nonmono and val_loss > min(best_val_loss[:-args.nonmono])):
                print('Switching!')
                optimizer = torch.optim.ASGD(model.parameters(), lr=args.lr, t0=0, lambd=0., weight_decay=args.wdecay)
                #optimizer.param_groups[0]['lr'] /= 2.
            best_val_loss.append(val_loss)
    # Run on test data.
    test_loss = evaluate(test_data, test_batch_size)
    print('=' * 89)
    print('| End of training | test loss {:5.2f} | test ppl {:8.2f}'.format(
        test_loss, math.exp(test_loss)))
    print('=' * 89)

except KeyboardInterrupt:
    print('-' * 89)
    print('Exiting from training early')



