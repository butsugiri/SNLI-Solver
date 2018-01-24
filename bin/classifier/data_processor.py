# -*- coding: utf-8 -*-

import json

import logzero
import numpy as np
from logzero import logger

from classifier import constants

DATA_TYPE = ('train', 'dev', 'test')


class DataProcessor(object):
    def __init__(self, log_name):
        # dictionary for vocabulary
        # each variable is set by calling load_vocab_from_path
        self.vocab = None
        self.ivocab = None

        self.logger = logger
        logzero.logfile(log_name)

    def load_data_from_path(self, data_path, data_type):
        assert data_type in DATA_TYPE

        # Vocabulary must be loaded in advance
        # (Otherwise the data cannot be converted to indices)
        assert self.vocab is not None
        self.logger.info('Loading {} data from {}'.format(data_type, data_path))

        dataset = self._make_dataset(data_path, self.vocab)
        self.logger.info('{} data is successfully loaded'.format(data_type))
        self.logger.info('{} data contains {} instances'.format(data_type, len(dataset)))
        return dataset

    def load_vocab_from_path(self, vocab_path):
        """
        Vocabulary file format: TOKEN_\t_INDEX
        """
        self.logger.info('Loading vocabulary from {}'.format(vocab_path))

        self.vocab = {x.strip().split('\t')[0]: int(x.strip().split()[1]) for x in open(vocab_path, 'r')}
        self.ivocab = {v: k for k, v in self.vocab.items()}

        assert constants.BOS_WORD in self.vocab
        assert constants.EOS_WORD in self.vocab
        assert constants.UNK_WORD in self.vocab
        assert self.vocab[constants.BOS_WORD] == constants.BOS
        assert self.vocab[constants.EOS_WORD] == constants.EOS
        assert self.vocab[constants.UNK_WORD] == constants.UNK

        self.logger.info('BOS token index: {}'.format(constants.BOS))
        self.logger.info('EOS token index: {}'.format(constants.EOS))
        self.logger.info('UNK token index: {}'.format(constants.UNK))

        self.logger.info('Vocabulary size: {}'.format(len(self.vocab)))
        self.logger.info('Vocabulary files are loaded')

    def _make_dataset(self, file_path, vocab):
        dataset = []
        with open(file_path, 'r') as input_data:
            for line in input_data:
                data = json.loads(line.strip())
                sent1_tokens = list(map(str.lower, data['sent1_tokens']))
                sent2_tokens = list(map(str.lower, data['sent2_tokens']))
                # TODO: posも読み込めるように
                sent1_pos = data['sent1_pos']
                sent2_pos = data['sent2_pos']
                sent1 = [vocab[t] if t in vocab else vocab[constants.UNK_WORD] for t in sent1_tokens]
                sent2 = [vocab[t] if t in vocab else vocab[constants.UNK_WORD] for t in sent2_tokens]
                if data['label'] == '-':
                    continue
                label = constants.Label2Index[data['label']]
                dataset.append((np.array(sent1, dtype='i'), np.array(sent2, dtype='i'), label))
        return dataset

    def dump_vocab_to(self, dest, kind='source'):
        assert kind == 'source' or kind == 'target'
        if kind == 'source':
            vocab2idx = self.src_vocab
        else:
            vocab2idx = self.trg_vocab

        with open(dest, 'w') as fo:
            for vocab, idx in vocab2idx.items():
                fo.write('{}\t{}\n'.format(vocab, idx))
            self.logger.info('{} vocabulary is saved at {}'.format(kind, dest))
