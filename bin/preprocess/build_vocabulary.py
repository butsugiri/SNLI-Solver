# -*- coding: utf-8 -*-
"""
"""

import argparse
import json
import os
import sys
from collections import defaultdict, Counter

sys.path.append('../classifier')
import constants


def build_vocabulary(file_path, limit, prefix, suffix, out_dir):
    vocab_count = Counter()
    with open(file_path, 'r') as fi:
        for line in fi:
            data = json.loads(line.strip())
            tokens = [x.lower() for x in data['sent1_tokens']] + [x.lower() for x in data['sent2_tokens']]
            for token in tokens:
                vocab_count[token] += 1

    word2id = defaultdict(lambda: len(word2id))
    word2id[constants.UNK_WORD]
    word2id[constants.BOS_WORD]
    word2id[constants.EOS_WORD]
    assert word2id[constants.UNK_WORD] == constants.UNK
    assert word2id[constants.BOS_WORD] == constants.BOS
    assert word2id[constants.EOS_WORD] == constants.EOS
    for token, count in vocab_count.most_common(limit):
        word2id[token]

    sys.stderr.write('Unknown word rate: {:.4f}% (= {} / {})'.format(len(word2id) * 100 / len(vocab_count), len(word2id), len(vocab_count)))
    file_name = '{}.{}.dict'.format(prefix, suffix)
    destination = os.path.join(out_dir, file_name)
    with open(destination, 'w') as fo:
        for token, index in sorted(word2id.items(), key=lambda x: x[1]):
            fo.write('{}\t{}\n'.format(token, index))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Vocabulary file builder")
    parser.add_argument('-v', '--vocab-limit', dest='limit', default=15000, type=int,
                        help='max source & target vocabulary size')
    parser.add_argument('--prefix', dest='prefix', default='vocabulary', type=str,
                        help='prefix')
    parser.add_argument('--suffix', dest='suffix', default='snli', type=str,
                        help='suffix')
    parser.add_argument('--lower', dest='lower', default=0, type=int, choices=[0, 1], help='lower sequence')
    parser.add_argument('--input', dest='input', required=True, type=str, help='input json')
    parser.add_argument('--out', dest='out', default='../../work/vocab', type=str, help='output dir')
    args = parser.parse_args()

    build_vocabulary(limit=args.limit, file_path=args.input, prefix=args.prefix, suffix='snli', out_dir=args.out)
