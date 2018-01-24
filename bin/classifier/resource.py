# -*- coding: utf-8 -*-
import os
import subprocess
import sys

import logzero
from logzero import logger
import chainer
import json
from datetime import datetime
import shutil


class Resource(object):
    def __init__(self, args, train=True):
        self.args = args
        self.logger = logger
        self.start_time = datetime.today()

        if train:
            self.output_dir = self._return_output_dir()
            self.create_output_dir()
            log_filename = 'train.log'
        else:
            self.output_dir = os.path.dirname(args.config_path)
            log_filename = 'inference.log'
        log_name = os.path.join(self.output_dir, log_filename)
        logzero.logfile(log_name)
        self.log_name = log_name
        self.logger.info('Log filename: [{}]'.format(log_name))

    def _return_output_dir(self):
        dir_name = '{}_seed_{}_optim_{}'.format(
            self.args.dir_prefix,
            self.args.seed,
            self.args.optimizer)
        output_dir = os.path.abspath(os.path.join(self.args.out, dir_name))
        return output_dir

    def create_output_dir(self):
        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)
            self.logger.info('Output Dir is created at [{}]'.format(self.output_dir))
        else:
            self.logger.info('Output Dir [{}] alreaady exists'.format(self.output_dir))

    def dump_git_info(self):
        if os.system('git rev-parse 2> /dev/null > /dev/null') == 0:
            self.logger.info('Git repository is found. Dumping logs & diffs...')
            git_log = '\n'.join(
                l for l in
                subprocess.check_output('git log --pretty=fuller | head -7', shell=True).decode('utf8').split('\n') if
                l)
            self.logger.info(git_log)

            git_diff = subprocess.check_output('git diff', shell=True).decode('utf8')
            self.logger.info(git_diff)
        else:
            self.logger.info('Git repository is not found. Continue...')

    def dump_command_info(self):
        self.logger.info('Command name: {}'.format(' '.join(sys.argv)))
        self.logger.info('Command is executed at: [{}]'.format(os.getcwd()))
        self.logger.info('Program is running at: [{}]'.format(os.uname().nodename))

    def dump_library_info(self):
        self.logger.info('Chainer Version: [{}]'.format(chainer.__version__))
        try:
            self.logger.info('CuPy Version: [{}]'.format(chainer.cuda.cupy.__version__))
        except AttributeError:
            self.logger.warn('CuPy was not found in your environment')

        if chainer.cuda.cudnn_enabled:
            self.logger.info('CuDNN is available')
        else:
            self.logger.warn('CuDNN is not available')

    def dump_python_info(self):
        self.logger.info('Python Version: [{}]'.format(sys.version))

    def save_config_file(self):
        with open(os.path.join(self.output_dir, 'config.json'), 'w') as fo:
            dumped_config = json.dumps(vars(self.args), sort_keys=True, indent=4)
            fo.write(dumped_config)
            self.logger.info('HyperParameters: {}'.format(dumped_config))

    def save_vocab_file(self):
        shutil.copy(self.args.vocab_path, self.output_dir)
        self.logger.info('Vocab file {} has been copied to {}'.format(self.args.vocab_path, self.output_dir))

    def dump_duration(self):
        end_time = datetime.today()
        self.logger.info('EXIT TIME: {}'.format(end_time.strftime('%Y%m%d - %H:%M:%S')))
        duration = end_time - self.start_time
        logger.info('Duration: {}'.format(str(duration)))
        logger.info('Remember: log is saved in {}'.format(self.output_dir))

