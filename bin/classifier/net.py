# -*- coding: utf-8 -*-
import chainer.functions as F
import chainer.links as L
from chainer import Chain
from chainer import reporter

from classifier.subfuncs import sequence_embed


class PartialSolver(Chain):
    def __init__(self, n_vocab, dim):
        super(PartialSolver, self).__init__()
        with self.init_scope():
            self.embed_mat = L.EmbedID(n_vocab, dim)
            self.bilstm = L.NStepBiLSTM(1, dim, int(dim / 2), dropout=0.2)
            self.l1 = L.Linear(dim, dim)
            self.l2 = L.Linear(dim, 3)

    def __call__(self, x1s, x2s):
        x2_out = self.encode_sequence(x2s)
        x2_out = F.dropout(x2_out, 0.2)
        logit = self.l2(F.dropout(F.relu(self.l1(x2_out)), 0.2))
        return logit

    def encode_sequence(self, xs):
        batch_size = len(xs)
        xs_embed = sequence_embed(self.embed_mat, xs)
        hsx, csx, hs = self.bilstm(None, None, xs_embed)
        return F.swapaxes(hsx, 0, 1).reshape(batch_size, -1)


class BidirectionalLSTM(Chain):
    def __init__(self, n_vocab, dim):
        super(BidirectionalLSTM, self).__init__()
        with self.init_scope():
            self.embed_mat = L.EmbedID(n_vocab, dim)
            self.bilstm = L.NStepBiLSTM(1, dim, int(dim / 2), dropout=0.2)
            self.l1 = L.Linear(dim * 2, dim)
            self.l2 = L.Linear(dim, 3)

    def __call__(self, x1s, x2s):
        x1_out = self.encode_sequence(x1s)
        x2_out = self.encode_sequence(x2s)
        logit = self.l2(F.relu(self.l1(F.concat([x1_out, x2_out], axis=1))))
        return logit

    def encode_sequence(self, xs):
        batch_size = len(xs)
        xs_embed = sequence_embed(self.embed_mat, xs)
        hsx, csx, hs = self.bilstm(None, None, xs_embed)
        return F.swapaxes(hsx, 0, 1).reshape(batch_size, -1)


class AveragedEmbedding(Chain):
    def __init__(self, n_vocab, dim):
        super(AveragedEmbedding, self).__init__()
        with self.init_scope():
            self.embed_mat = L.EmbedID(n_vocab, dim)
            self.l1 = L.Linear(dim * 2, dim)
            self.l2 = L.Linear(dim, 3)

    def __call__(self, x1s, x2s):
        x1_len = self.xp.array([len(x) for x in x1s], dtype='i')[:, None]
        x1_embed = sequence_embed(self.embed_mat, x1s)
        x1_sum = F.sum(F.stack(F.pad_sequence(x1_embed, padding=0)), axis=1)
        x1_avg = x1_sum / x1_len

        x2_len = self.xp.array([len(x) for x in x2s], dtype='i')[:, None]
        x2_embed = sequence_embed(self.embed_mat, x2s)
        x2_sum = F.sum(F.stack(F.pad_sequence(x2_embed, padding=0)), axis=1)
        x2_avg = x2_sum / x2_len

        logit = self.l2(F.relu(self.l1(F.concat([x1_avg, x2_avg]))))
        return logit


class TextClassification(Chain):
    def __init__(self, n_vocab, dim):
        super(TextClassification, self).__init__()
        with self.init_scope():
            if True:
                self.model = PartialSolver(n_vocab, dim)
            else:
                self.model = None

    def __call__(self, x1s, x2s):
        return self.model(x1s, x2s)

    def compute_loss(self, x1s, x2s, t):
        pred = self(x1s, x2s)
        loss = F.softmax_cross_entropy(pred, t, ignore_label=-1)
        reporter.report({'loss': loss.data}, self)

        acc = self.compute_accuracy(pred, t)
        reporter.report({'acc': acc}, self)
        return loss

    @staticmethod
    def compute_accuracy(pred, target):
        pred_label = pred.data.argmax(axis=1)
        acc = int(sum(pred_label == target)) / target.size
        return acc
