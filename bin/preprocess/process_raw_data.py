# -*- coding: utf-8 -*-
"""

"""
import json
import sys

from nltk.tree import Tree


def parse_s_expression(s_exp):
    pos_seq = [s.label() for s in Tree.fromstring(s_exp).subtrees(lambda t: t.height() == 2)]
    token_seq = [s.leaves()[0] for s in Tree.fromstring(s_exp).subtrees(lambda t: t.height() == 2)]
    try:
        assert len(pos_seq) == len(token_seq)
    except AssertionError:
        print(pos_seq)
        print(token_seq)
    return pos_seq, token_seq


def main(fi):
    for line in fi:
        data = json.loads(line.strip())
        label = data['gold_label']
        sent1_pos, sent1_tokens = parse_s_expression(data['sentence1_parse'])
        sent2_pos, sent2_tokens = parse_s_expression(data['sentence2_parse'])
        out = {
            'label': label,
            'sent1_pos': sent1_pos,
            'sent1_tokens': sent1_tokens,
            'sent2_pos': sent2_pos,
            'sent2_tokens': sent2_tokens
        }
        print(json.dumps(out))


if __name__ == "__main__":
    main(sys.stdin)
