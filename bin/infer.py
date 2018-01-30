# -*- coding: utf-8 -*-
import argparse

import chainer
import numpy as np

from classifier.data_processor import DataProcessor
from classifier.iterator import BucketIterator
from classifier.net import TextClassification
from classifier.resource import Resource
from classifier.subfuncs import convert


def main():
    parser = argparse.ArgumentParser(description='SNLI Solver',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--batchsize', '-b', type=int, default=32, help='Number of Sentences in Each Mini-Batch')
    parser.add_argument('--gpu', '-g', type=int, default=-1, help='GPU ID (Negative Value Indicates CPU)')

    # Arguments for the dataset / vocabulary path
    parser.add_argument('--out', dest='out', required=True, type=str, help='')
    parser.add_argument('--epoch', dest='epoch', required=True, type=int, help='')
    parser.add_argument('--input-file', dest='input_file_path', required=True, type=str, help='')
    args = parser.parse_args()

    resource = Resource(args, train=False)
    logger = resource.logger

    resource.load_config()
    vocab_path = resource.get_vocab_path()
    model_path = resource.get_model_path()

    dataset = DataProcessor(resource.log_name)
    dataset.load_vocab_from_path(vocab_path)
    test_data = dataset.load_data_from_path(args.input_file_path, 'test')
    test_iter = BucketIterator(dataset=test_data, batch_size=args.batchsize, shuffle=False, debug=False, repeat=False)
    model = TextClassification(n_vocab=len(dataset.vocab), dim=resource.config['embed_dim'])

    chainer.serializers.load_npz(model_path, model)

    if args.gpu >= 0:
        chainer.cuda.get_device_from_id(args.gpu).use()
        model.to_gpu(args.gpu)

    target_list = []
    pred_list = []
    for batch in test_iter:
        conv_batch = convert(batch, args.gpu)
        x1s = conv_batch['x1s']
        x2s = conv_batch['x2s']
        target = conv_batch['t'].tolist()
        pred = model(x1s, x2s).data.argmax(axis=1).tolist()

        target_list += target
        pred_list += pred

    acc = np.sum(np.array(target_list) == np.array(pred_list)) / len(target_list)
    print(acc)


if __name__ == "__main__":
    main()
