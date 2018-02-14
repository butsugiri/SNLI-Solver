# -*- coding: utf-8 -*-
import chainer.functions as F
import chainer.links as L
import numpy as np
from chainer import Chain
from chainer import reporter


def sequence_embed(embed, xs):
    """
    Embedding層を効率よく潜らせるための関数
    :param embed: L.EmbedID
    :param xs: [[x1, x2,..., xn]] * batch_size
    :return: [[ex1, ex2, ex3,..., exn]] * batch_size
    """
    xs_len = [len(x) for x in xs]
    x_section = np.cumsum(xs_len[:-1])
    ex = embed(F.concat(xs, axis=0))
    exs = F.split_axis(ex, x_section, 0)
    return exs


def calc_vector_interactions(xs, xs_dash):
    """
    for given two vectors, calculate:
    1. Subtraction
    2. Element-wise multiplication
    and then concatenate them
    """
    return F.concat([xs, xs_dash, xs - xs_dash, xs * xs_dash], axis=2)


def average_pooling(xs, xs_len):
    xs_pad = F.pad_sequence(xs, padding=0)  # これで可変長系列の平均を計算
    return F.sum(xs_pad, axis=1) / xs_len[:, None]


def max_pooling(xs):
    xs_pad = F.pad_sequence(xs, padding=-np.inf)  # これで可変長系列のmax-poolingを計算
    return F.max(xs_pad, axis=1)


class InputEncodingLayer(Chain):
    def __init__(self, n_vocab, embed_dim, hidden_dim):
        super(InputEncodingLayer, self).__init__()
        with self.init_scope():
            self.embed_mat = L.EmbedID(n_vocab, embed_dim)
            self.hypo_encoder = L.NStepBiLSTM(1, hidden_dim, int(hidden_dim / 2), dropout=0.0)
            self.premise_encoder = L.NStepBiLSTM(1, hidden_dim, int(hidden_dim / 2), dropout=0.0)

    def __call__(self, x1s, x2s):
        h1s = self.encode_sequence(x1s, self.hypo_encoder)
        h2s = self.encode_sequence(x2s, self.premise_encoder)
        return h1s, h2s

    def encode_sequence(self, xs, encoder):
        xs_embed = sequence_embed(self.embed_mat, xs)
        _, _, hs = encoder(None, None, xs_embed)
        return hs


class LocalInferenceLayer(Chain):
    def __init__(self):
        super(LocalInferenceLayer, self).__init__()
        with self.init_scope():
            pass  # 特に管理するべきパラメータは存在しない

    def __call__(self, h1s, h2s):
        # 散らかってるが，とりあえずコレで
        seq_len, _ = h1s[0].shape
        h2s_len = [x.shape[0] for x in h2s]

        h1s_stack = F.stack(h1s, axis=0)
        h2s_stack = F.pad_sequence(h2s, padding=-1)

        h2s_mask = self.xp.swapaxes((h2s_stack.data != -1)[:, :, :seq_len], 1, 2)
        minfs = self.xp.full(h2s_mask.shape, -np.inf, dtype=np.float32)
        raw_attn_mat = F.batch_matmul(h1s_stack, F.swapaxes(h2s_stack, 1, 2))
        masked_attn_mat = F.where(h2s_mask, raw_attn_mat, minfs)

        # h1s 方向に重み付き和を計算
        h1s_attn = F.batch_matmul(F.softmax(masked_attn_mat, axis=2), h2s_stack)
        h1s = F.separate(calc_vector_interactions(h1s_stack, h1s_attn), axis=0)

        # h2s 方向に重み付き和を計算
        h2s_attn_mat = F.softmax(masked_attn_mat, axis=1)
        # こっちの方向だと，softmax計算時にnanが生まれるので，それを0埋め
        # 0埋めしないとnanと実数との積が発生し，全体の計算が死んでしまう
        masked_h2s_attn_mat = F.where(h2s_mask, h2s_attn_mat, self.xp.zeros(h2s_mask.shape, dtype='f'))
        h2s_attn = F.swapaxes(F.batch_matmul(F.swapaxes(h1s_stack, 1, 2), masked_h2s_attn_mat), 1, 2)
        h2s = [h[:l, :] for h, l in zip(F.separate(calc_vector_interactions(h2s_stack, h2s_attn), axis=0), h2s_len)]
        return h1s, h2s


class OutputLayer(Chain):
    def __init__(self, in_dim, out_dim, dropout_rate):
        super(OutputLayer, self).__init__()
        self.dropout_rate = dropout_rate
        with self.init_scope():
            self.l1 = L.Linear(in_dim, in_dim)
            self.l2 = L.Linear(in_dim, out_dim)

    def __call__(self, h):
        return self.l2(F.dropout(F.tanh(self.l1(F.dropout(h, self.dropout_rate))), self.dropout_rate))


class InferenceCompositionLayer(Chain):
    def __init__(self, hidden_dim):
        super(InferenceCompositionLayer, self).__init__()
        with self.init_scope():
            self.hypo_encoder = L.NStepBiLSTM(1, hidden_dim, int(hidden_dim / 8), dropout=0.0)
            self.premise_encoder = L.NStepBiLSTM(1, hidden_dim, int(hidden_dim / 8), dropout=0.0)

    def __call__(self, m1s, m2s):
        m1s_len = self.xp.array([m.shape[0] for m in m1s], dtype='f')  # 本来なら必要ないが
        m2s_len = self.xp.array([m.shape[0] for m in m2s], dtype='f')
        _, _, v1s = self.hypo_encoder(None, None, m1s)
        _, _, v2s = self.premise_encoder(None, None, m2s)

        v1s_avg = average_pooling(v1s, m1s_len)
        v1s_max = max_pooling(v1s)
        v2s_avg = average_pooling(v2s, m2s_len)
        v2s_max = max_pooling(v2s)

        ret = F.concat([v1s_avg, v1s_max, v2s_avg, v2s_max], axis=1)
        return ret


class EnhancedSequentialInferenceModel(Chain):
    """
    Implementation of ESIM: http://www.aclweb.org/anthology/P17-1152
    """
    def __init__(self, n_vocab, embed_dim, hidden_dim, dropout_rate):
        super(EnhancedSequentialInferenceModel, self).__init__()
        with self.init_scope():
            self.input_encoding = InputEncodingLayer(n_vocab, embed_dim, hidden_dim)
            self.local_inference = LocalInferenceLayer()
            self.inference_composition = InferenceCompositionLayer(hidden_dim * 4)
            self.output = OutputLayer(hidden_dim * 4, out_dim=3, dropout_rate=dropout_rate)

    def __call__(self, x1s, x2s):
        h1s, h2s = self.input_encoding(x1s, x2s)
        m1s, m2s = self.local_inference(h1s, h2s)
        v = self.inference_composition(m1s, m2s)
        logit = self.output(v)
        return logit

    def compute_loss(self, x1s, x2s, t):
        logit = self(x1s, x2s)
        loss = F.softmax_cross_entropy(logit, t, ignore_label=-1)
        reporter.report({'loss': loss.data}, self)

        acc = self.compute_accuracy(logit, t)
        reporter.report({'acc': acc}, self)
        return loss

    @staticmethod
    def compute_accuracy(pred, target):
        pred_label = pred.data.argmax(axis=1)
        acc = int(sum(pred_label == target)) / target.size
        return acc
