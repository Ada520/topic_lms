from torch.autograd import Variable


def repackage_hidden(h):
    """Wraps hidden states in new Variables, to detach them from their history."""
    if type(h) == Variable:
        return Variable(h.data)
    else:
        return tuple(repackage_hidden(v) for v in h)


def batchify(data, args):
    # Work out how cleanly we can divide the dataset into bsz parts.
    nbatch = data.size(0) // args.batch_size
    # Trim off any extra elements that wouldn't cleanly fit (remainders).
    data = data.narrow(0, 0, nbatch * args.batch_size)
    # Evenly divide the data across the bsz batches.
    data = data.view(args.batch_size, -1).t().contiguous()
    if args.cuda:
        return data.cuda()
    else:
        return data

# def get_batch(source, i, args, seq_len=None, evaluation=False):
#     seq_len = min(seq_len if seq_len else args.bptt, len(source) - 1 - i)
#     data = Variable(source[i:i+seq_len], volatile=evaluation)
#     target = Variable(source[i+1:i+1+seq_len].view(-1))
#     return data, target
def get_batch(source, i, args):
    seq_len = min(args.bptt, len(source) - 1 - i)
    data = source[i:i+seq_len]
    target = source[i+1:i+1+seq_len].view(-1)
    return data, target
