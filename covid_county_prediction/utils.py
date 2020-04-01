import torch, os, matplotlib
import numpy as np
from collections import defaultdict
import covid_county_prediction.constants as constants

PLOT_FLAG_MAP = {
    'continuous': '-',
    'discrete': 'o'
}
POINT_FLIP_THRESHOLD = 10000

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
        print('\t'.join(entries))

    def _get_batch_fmtstr(self, num_batches):
        num_digits = len(str(num_batches // 1))
        fmt = '{:' + str(num_digits) + 'd}'
        return '[' + fmt + '/' + fmt.format(num_batches) + ']'
