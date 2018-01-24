# -*- coding: utf-8 -*-
import random
from collections import defaultdict

import numpy as np
from chainer.iterators import SerialIterator


class BucketIterator(SerialIterator):
    """
    This iterator sorts dataset according to the length of source/target sentences,
    and separates them into the 'bucket' of sentences of the same lengths.
    Thus, the mini-batch contains (source) sentences of same lengths.
    """

    def __init__(self, dataset, batch_size, repeat=True, shuffle=True, sort_by='source', debug=False):
        super().__init__(
            dataset, batch_size, repeat=repeat, shuffle=shuffle
        )
        assert sort_by == 'source' or sort_by == 'target'
        self.debug = debug

        # Create a bucket according to the source sent length
        self.buckets = defaultdict(list)
        for src, trg, *x in self.dataset:
            sort_key = src if sort_by == 'source' else trg
            self.buckets[len(sort_key)].append((src, trg, *x))
        self.create_order()

    def create_order(self):
        self._order = []
        for lengths, datasets in self.buckets.items():
            if self._shuffle:
                np.random.shuffle(datasets)
            indices = range(0, len(datasets), self.batch_size)
            self._order.extend([datasets[_:_ + self.batch_size] for _ in indices])

        if self._shuffle:
            np.random.shuffle(self._order)

        if self.debug:
            self._order = sorted(self._order, key=lambda x: len(x[0][0]), reverse=True)

    @property
    def epoch_detail(self):
        return self.epoch + self.current_position / len(self._order)

    def __next__(self):
        if not self._repeat and self.epoch > 0:
            raise StopIteration

        self._previous_epoch_detail = self.epoch_detail
        N = len(self._order)

        batch = self._order[self.current_position]

        if self.current_position + 1 >= N:
            self.current_position = 0
            self.epoch += 1
            self.is_new_epoch = True
            self.create_order()
        else:
            self.is_new_epoch = False
            self.current_position += 1

        return batch

    # Iterator just won't work without this
    next = __next__

    def reset(self):
        self.current_position = 0
        self.epoch = 0
        self.is_new_epoch = False

        # use -1 instead of None internally.
        self._previous_epoch_detail = -1.

    def serialize(self, serializer):
        self.current_position = serializer('current_position',
                                           self.current_position)
        self.epoch = serializer('epoch', self.epoch)
        self.is_new_epoch = serializer('is_new_epoch', self.is_new_epoch)
        if self._order is not None:
            try:
                serializer('order', self._order)
            except KeyError:
                serializer('_order', self._order)
        try:
            self._previous_epoch_detail = serializer(
                'previous_epoch_detail', self._previous_epoch_detail)
        except KeyError:
            # guess previous_epoch_detail for older version
            self._previous_epoch_detail = self.epoch + \
                                          (self.current_position - self.batch_size) / len(self.dataset)
            if self.epoch_detail > 0:
                self._previous_epoch_detail = max(
                    self._previous_epoch_detail, 0.)
            else:
                self._previous_epoch_detail = -1.


class SortedSerialIterator(SerialIterator):
    # TODO: consider the case shuffle=False
    def __init__(self, dataset, batch_size, repeat=True, shuffle=True):
        super().__init__(
            dataset, batch_size, repeat=repeat, shuffle=shuffle
        )

        # Create a bucket according to the source sent length
        self.buckets = defaultdict(list)
        for n, (data) in enumerate(self.dataset):
            src, trg, *_ = data
            self.buckets[len(src)].append(n)

        self.create_order()

    @property
    def epoch_detail(self):
        return self.epoch + self.current_position / len(self._order)

    def create_order(self):
        # Shuffle sentence with same lengths
        if self._shuffle:
            for idx in self.buckets.values():
                np.random.shuffle(idx)

        indices = np.concatenate([self.buckets[x] for x in sorted(self.buckets)])
        assert len(indices) == len(self.dataset)

        batches = [indices[i:i + self.batch_size] for i in range(0, len(indices), self.batch_size)]
        fully_filled_batch = batches[:-1]
        rest_batch = batches[-1]

        self._order = []
        split_position = range(0, len(self.dataset), self.batch_size)
        if self._shuffle:
            indices = np.concatenate([np.concatenate(fully_filled_batch), rest_batch])
            xs = [indices[i:i + self.batch_size] for i in split_position]
            for x in xs:
                self._order.append([self.dataset[i] for i in x])
            # ここでミニバッチの順番をシャッフル
            random.shuffle(self._order)
        else:
            # シャッフルしないなら
            # 順番に先頭からバッチを作る
            for i in split_position:
                self._order.append(self.dataset[i:i + self.batch_size])

    def __next__(self):
        if not self._repeat and self.epoch > 0:
            raise StopIteration

        self._previous_epoch_detail = self.epoch_detail

        i = self.current_position
        i_end = i + 1
        N = len(self._order)

        batch = self._order[i]

        if i_end >= N:
            if self._repeat:
                if self._shuffle:
                    self.create_order()
            self.current_position = 0
            self.epoch += 1
            self.is_new_epoch = True
        else:
            self.is_new_epoch = False
            self.current_position = i_end

        return batch

    next = __next__

    def reset(self):
        self.current_position = 0
        self.epoch = 0
        self.is_new_epoch = False

        # use -1 instead of None internally.
        self._previous_epoch_detail = -1.
