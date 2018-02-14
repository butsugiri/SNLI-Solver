# -*- coding: utf-8 -*-
"""
Parameter initialization/print/registration/comparison
"""
import os

import chainer
import logzero
import numpy as np
from chainer import cuda
from logzero import logger


# from EncoderDecoder import get_module_logger


class ParameterHelper(object):
    def __init__(self, log_name):
        logzero.logfile(log_name)
        self.logger = logger
        self.registered_params = None

    def print_params(self, optimizer):
        """
        print parameters of given optimizer
        :param optimizer: chainer optimizer
        :return: None
        """
        total_norm = 0
        total_param = 0
        named_params = sorted(optimizer.target.namedparams(), key=lambda x: x[0])
        self.logger.info('Printing all parametes...')
        for name, param in named_params:
            t_norm = chainer.optimizer._sum_sqnorm(param.data)
            self.logger.info(
                'Name: {name} Ndim: {ndim} Shape: {shape} Size: {size} Norm: {norm}'.format(
                    name=name,
                    ndim=param.data.ndim,
                    shape=param.data.shape,
                    size=param.data.size,
                    norm=t_norm))
            total_norm += t_norm
            total_param += param.data.size

        xp = cuda.cupy if chainer.cuda.available else np
        self.logger.info('Total param size = {}'.format(total_param))
        self.logger.info('Total norm size = {}'.format(xp.sqrt(total_norm)))
        self.logger.info('Printed all parameters')

    def load_glove_vector(self, glove_path, vocab, embed_mat):
        embed_array = np.random.normal(scale=0.05, size=embed_mat.W.shape).astype('f')
        self.logger.info('loading pretrained word vector from [{}]'.format(os.path.abspath(glove_path)))
        with open(glove_path, 'r') as fi:
            for line in fi:
                word, *vec = line.strip().split()
                if word in vocab:
                    embed_array[vocab[word]] = np.array(vec, dtype='f')
        embed_mat.W.data = embed_array

    def initialize_params(self, optimizer, init_type, init_scale):
        """
        initialize params of given optimizer
        :param optimizer: chainer optimizer
        :param init_type:
        :param init_scale:
        :return:
        """
        self.logger.info('The initializer is [{}] with scale [{}]'.format(init_type, init_scale))
        if init_type == 'uniform':
            initializer = chainer.initializers.Uniform(init_scale)
        elif init_type == 'normal':
            initializer = chainer.initializers.Normal(init_scale)
        elif init_type == 'default':  # chainer default
            initializer = None
        else:
            raise NotImplementedError

        if initializer:
            # 最初にソートしておかないと，optimizer.target.params()の順番は呼ぶたびに違うので，
            # 違うパラメータの初期化がなされる
            named_params = sorted(optimizer.target.namedparams(), key=lambda x: x[0])
            for name, param in named_params:
                with cuda.get_device(param.data):
                    param.copydata(chainer.Parameter(initializer, param.data.shape))
        else:
            # let initializers do their default initialization
            pass
        self.logger.info('Finished Parameter Initialization')

    def register_params(self, optimizer):
        named_params = sorted(optimizer.target.namedparams(), key=lambda x: x[0])
        self.registered_params = {}
        for name, param in named_params:
            norm = chainer.optimizer._sum_sqnorm(param.data)
            self.registered_params[name] = norm

    def compare_params(self, optimizer):
        """
        Compare registered params and optimizer params
        mainly for debugging purposes, to check that
        pretrained part of the model is properly initialized.
        :param optimizer: chainer optimizer
        :return:
        """
        assert self.registered_params is not None
        self.logger.info('Comparing norm values...')

        named_params = sorted(optimizer.target.namedparams(), key=lambda x: x[0])
        for name, param in named_params:
            norm = chainer.optimizer._sum_sqnorm(param.data)
            if name in self.registered_params:
                out = '{}:\t{} --> {}'.format(name, self.registered_params[name], norm)
            else:
                out = '{} with norm {} is not found in registered params'.format(name, norm)
            self.logger.info(out)
