# -*- coding: utf-8 -*-
import argparse

import chainer
from chainer import training
from chainer.training import extensions

from classifier.data_processor import DataProcessor
from classifier.iterator import BucketIterator
from classifier.net import EnhancedSequentialInferenceModel
from classifier.params import ParameterHelper
from classifier.resource import Resource
from classifier.subfuncs import convert
from classifier.subfuncs import set_random_seed


def main():
    parser = argparse.ArgumentParser(description='ESIM',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument('--batchsize', '-b', type=int, default=32,
                        help='Number of sentences in each mini-batch')
    parser.add_argument('--epoch', '-e', type=int, default=13,
                        help='Number of sweeps over the dataset to train')
    parser.add_argument('--gpu', '-g', type=int, default=-1,
                        help='GPU ID (negative value indicates CPU)')
    parser.add_argument('--resume', '-r', default='',
                        help='Resume the training from snapshot')
    parser.add_argument('--hidden-dim' '-H', dest='hidden_dim', type=int, default=300,
                        help='Size of hidden dimension')
    parser.add_argument('--embed-dim' '-E', dest='embed_dim', type=int, default=300,
                        help='Size of embed dimension')
    parser.add_argument('--optimizer', '-O', dest='optimizer', type=str, default='Adam',
                        choices=['Adam', 'SGD'], help='Type of optimizer')
    parser.add_argument('--model-type', dest='model_type', type=str, default='BiLSTM',
                        choices=['BiLSTM', 'Partial', 'Embed'], help='Model Type')
    parser.add_argument('--dropout', dest='dropout_rate', type=float, default=0.5, help='dropout rate')
    parser.add_argument('--drop-local-inference', dest='drop_local_inference', type=int, choices=[0, 1], default=0,
                        help='Use dropout for the input of local inference layer')
    parser.add_argument('--learning-rate', '--lr', dest='learning_rate', type=float, default=0.0004,
                        help='learning rate')
    parser.add_argument('--initializer-scale', dest='init_scale', default=0.1, type=float,
                        help='Scaling factor for parameter initializer')
    parser.add_argument('--initializer-type', dest='init_type', default='uniform',  # same as OpenNMT
                        help='select initializer [string] default=uniform option:chainer_default, normal')

    # Arguments for the dataset / vocabulary path
    parser.add_argument('--vocab', dest='vocab_path', required=True,
                        help='Path to the vocabulary file')
    parser.add_argument('--train-data-file', dest='train_data_path', required=True,
                        help='Path to the train data')
    parser.add_argument('--dev-data-file', dest='dev_data_path', required=True,
                        help='Path to the development data')
    parser.add_argument('--glove-path', dest='glove_path', default='',
                        help='Path to the pretrained word vector data')

    parser.add_argument('--seed', default=0, type=int, help='Seed for Random Module')

    # Arguments for directory
    parser.add_argument('--out', '-o', default='../result', help='Directory to output the result')
    parser.add_argument('--dir-prefix', dest='dir_prefix', default='model', type=str, help='Prefix of the output dir')
    args = parser.parse_args()
    set_random_seed(args.seed, args.gpu)

    resource = Resource(args, train=True)
    resource.dump_git_info()
    resource.dump_command_info()
    resource.dump_python_info()
    resource.dump_library_info()
    resource.save_vocab_file()
    resource.save_config_file()

    logger = resource.logger

    dataset = DataProcessor(resource.log_name)
    dataset.load_vocab_from_path(args.vocab_path)
    train_data = dataset.load_data_from_path(args.train_data_path, 'train')
    valid_data = dataset.load_data_from_path(args.dev_data_path, 'dev')
    model = EnhancedSequentialInferenceModel(
        n_vocab=len(dataset.vocab),
        embed_dim=args.embed_dim,
        hidden_dim=args.hidden_dim,
        dropout_rate=args.dropout_rate,
        drop_local_inference=args.drop_local_inference
    )

    if args.optimizer == 'Adam':
        optimizer = chainer.optimizers.Adam(alpha=args.learning_rate)
    else:
        optimizer = chainer.optimizers.SGD(lr=args.learning_rate)
    optimizer.setup(model)
    logger.info('Optimizer is set to [{}]'.format(args.optimizer))

    param_helper = ParameterHelper(resource.log_name)
    param_helper.initialize_params(optimizer, init_type=args.init_type, init_scale=args.init_scale)
    if args.glove_path != '':
        param_helper.load_glove_vector(args.glove_path, dataset.vocab, embed_mat=model.input_encoding.embed_mat)

    # Send model to GPU (according to the arguments)
    if args.gpu >= 0:
        model.to_gpu(args.gpu)

    train_iter = BucketIterator(dataset=train_data, batch_size=args.batchsize, shuffle=True, debug=False)
    updater = training.updater.StandardUpdater(train_iter, optimizer, converter=convert, device=args.gpu,
                                               loss_func=model.compute_loss)
    trainer = training.Trainer(updater, (args.epoch, 'epoch'), out=resource.output_dir)

    # if args.resume:
    #     logger.info('Loading trainer snapshot file {}'.format(args.resume))
    #     load_npz(args.resume, trainer, strict=False)
    #     logger.info('Snapshot has been loaded successfully')

    # TrainerのExtension追加セクション
    short_term = (200, 'iteration')
    long_term = (1, 'epoch')

    dev_iter = BucketIterator(valid_data, args.batchsize, repeat=False)
    trainer.extend(
        extensions.Evaluator(dev_iter, model, device=args.gpu, converter=convert, eval_func=model.compute_loss),
        trigger=long_term)
    trainer.extend(extensions.ProgressBar(update_interval=1))
    trainer.extend(extensions.LogReport(trigger=short_term, log_name='chainer_report_iteration.log'),
                   trigger=short_term, name='iteration')
    trainer.extend(extensions.LogReport(trigger=long_term, log_name='chainer_report_epoch.log'), trigger=long_term,
                   name='epoch')
    trainer.extend(extensions.snapshot_object(model, 'model_epoch_{.updater.epoch}'), trigger=long_term)
    trainer.extend(extensions.snapshot_object(optimizer, 'optim_epoch_{.updater.epoch}'), trigger=long_term)

    entries = ['epoch', 'iteration', 'main/loss', 'validation/main/loss', 'main/acc', 'validation/main/acc']
    trainer.extend(extensions.PrintReport(entries=entries, log_report='iteration'), trigger=short_term)
    trainer.extend(extensions.PrintReport(entries=entries, log_report='epoch'), trigger=long_term)

    logger.info('Start training...')
    trainer.run()
    logger.info('Training complete!!')
    resource.dump_duration()


if __name__ == "__main__":
    main()
