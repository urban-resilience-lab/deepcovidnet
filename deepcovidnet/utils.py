import logging
from time import time


class AverageMeter(object):
    """Computes and stores the average and current value"""
    #taken from https://github.com/pytorch/examples/blob/master/imagenet/main.py
    def __init__(self, name, fmt=':6.3f'):
        self.name = name
        self.fmt = fmt
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def __str__(self):
        fmtstr = '{name} {val' + self.fmt + '} ({avg' + self.fmt + '})'
        return fmtstr.format(**self.__dict__)


class ProgressMeter(object):
    #taken from https://github.com/pytorch/examples/blob/master/imagenet/main.py
    def __init__(self, num_batches, meters, prefix=""):
        self.batch_fmtstr = self._get_batch_fmtstr(num_batches)
        self.meters = meters
        self.prefix = prefix

    def display(self, batch, epoch):
        entries = [self.prefix + ' ' + str(epoch) + ':' + self.batch_fmtstr.format(batch)]
        entries += [str(meter) for meter in self.meters]
        print('\t'.join(entries), end='\n\n')

    def _get_batch_fmtstr(self, num_batches):
        num_digits = len(str(num_batches // 1))
        fmt = '{:' + str(num_digits) + 'd}'
        return '[' + fmt + '/' + fmt.format(num_batches) + ']'


def timed_logger_decorator(f):
    def wrapper(*args, **kwargs):
        logging.info(f'Entering {f.__name__}')
        t = time()
        ans = f(*args, **kwargs)
        t = time() - t
        logging.info(f'Exiting {f.__name__} after {t} secs')
        return ans

    return wrapper
