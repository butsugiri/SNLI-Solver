# -*- coding: utf-8 -*-

import random

import chainer
import chainer.functions as F
import numpy as np
from chainer import cuda
from chainer.dataset import to_device


def set_random_seed(seed, gpu):
    # set Python random seed
    random.seed(seed)
    # set NumPy random seed
    np.random.seed(seed)
    # set CuPy random seed
    if gpu >= 0:
        chainer.cuda.get_device_from_id(gpu).use()
        chainer.cuda.cupy.random.seed(seed)


def sequence_embed(embed, xs):
    """
    Embedding層を効率よく潜らせるための関数
    :param embed: L.EmbedID
    :param xs: [[x1, x2,..., xn]] * batch_size
    :return: [[ex1, ex2, ex3,..., exn]] * batch_size
    """
    x_len = [len(x) for x in xs]
    x_section = np.cumsum(x_len[:-1])
    ex_original = embed(F.concat(xs, axis=0))
    exs = F.split_axis(ex_original, x_section, 0)
    return exs


def convert(batch, gpu):
    """
    """

    def to_device_batch(batch, gpu):
        if gpu is None:
            return batch
        elif gpu < 0:
            return [chainer.dataset.to_device(gpu, x) for x in batch]
        else:
            xp = cuda.cupy.get_array_module(*batch)
            concat = xp.concatenate(batch, axis=0)
            sections = np.cumsum([len(x) for x in batch[:-1]], dtype='i')
            concat_dev = chainer.dataset.to_device(gpu, concat)
            batch_dev = cuda.cupy.split(concat_dev, sections)
            return batch_dev

    return {'x1s': to_device_batch([x1 for x1, x2, t in batch], gpu=gpu),
            'x2s': to_device_batch([x2 for x1, x2, t in batch], gpu=gpu),
            't': to_device(x=np.array([t for _, _, t in batch], dtype='i'), device=gpu)
            }


# strict=False オプションを使うためだけの関数
# Chainer v2.0.1時点ではまだマージされていない
def load_npz(filename, obj, strict=False):
    with np.load(filename) as f:
        d = chainer.serializers.NpzDeserializer(f, strict=strict)
        d.load(obj)


class ThresholdTrigger(object):
    """
    The trigger that activates after certain epoch
    e.g. Keep the learning rate fixed for first n epochs,
    and then decay it by 0.95 for each epoch
    """

    def __init__(self, period, unit, threshold):
        self.period = period
        assert unit == 'epoch' or unit == 'iteration'
        self.unit = unit
        self.count = 0
        self.threshold = threshold

    def __call__(self, trainer):
        updater = trainer.updater
        if self.unit == 'epoch':
            prev = self.count
            self.count = updater.epoch_detail // self.period
            return (prev != self.count) and (self.count > self.threshold)
        else:
            iteration = updater.iteration
            return iteration > 0 and iteration % self.period == 0
