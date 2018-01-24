# -*- coding: utf-8 -*-
import chainer.functions as F
import chainer.links as L
from chainer import Chain
from chainer import reporter

from classifier.subfuncs import sequence_embed


class TextClassification(Chain):
    def __init__(self, n_vocab, dim):
        super(TextClassification, self).__init__()
        with self.init_scope():
            self.embed_mat = L.EmbedID(n_vocab, dim)
            self.liner_1 = L.Linear(dim * 2, dim)
            self.liner_2 = L.Linear(dim, 3)

    def __call__(self, x1s, x2s):
        x1_len = self.xp.array([len(x) for x in x1s], dtype='i')[:, None]
        x1_embed = sequence_embed(self.embed_mat, x1s)
        x1_sum = F.sum(F.stack(F.pad_sequence(x1_embed, padding=0)), axis=1)
        x1_avg = x1_sum / x1_len

        x2_len = self.xp.array([len(x) for x in x2s], dtype='i')[:, None]
        x2_embed = sequence_embed(self.embed_mat, x2s)
        x2_sum = F.sum(F.stack(F.pad_sequence(x2_embed, padding=0)), axis=1)
        x2_avg = x2_sum / x2_len

        pred = self.liner_2(F.relu(self.liner_1(F.concat([x1_avg, x2_avg]))))
        return pred

    def compute_loss(self, x1s, x2s, t):
        pred = self(x1s, x2s)
        loss = F.softmax_cross_entropy(pred, t, ignore_label=-1)
        reporter.report({'loss': loss.data}, self)

        acc = self.compute_accuracy(pred, t)
        reporter.report({'acc': acc}, self)
        return loss

    def compute_accuracy(self, pred, target):
        pred_label = pred.data.argmax(axis=1)
        acc = int(sum(pred_label == target)) / target.size
        return acc
