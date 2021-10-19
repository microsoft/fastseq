# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""BERT Dictionary"""

from collections import Counter
from multiprocessing import Pool
import logging
import os
import torch

from fairseq.tokenizer import tokenize_line
from fairseq.binarizer import safe_readline
from fairseq.data import data_utils, Dictionary
from fastseq.logging import get_logger

logger = get_logger(__name__, logging.INFO)

class BertDictionary(Dictionary):
    """A mapping from symbols to consecutive integers"""

    def __init__(
        self,
        pad='<pad>',
        eos='</s>',
        unk='<unk>',
        bos='<s>',
        extra_special_symbols=None,
    ):
        super().__init__(pad=pad, eos=eos, unk=unk, bos=bos, extra_special_symbols=extra_special_symbols)

    @classmethod
    def load_from_file(cls, filename):
        """Load the dictionary from file"""
        d = cls()
        d.symbols = []
        d.count = []
        d.indices = {}

        line_cnt = 0
        with open(
            filename, 'r', encoding='utf-8', errors='ignore') as input_file:
            for line in input_file:
                line_cnt += 1
                try:
                    k, v = line.split(" ")
                    d.add_symbol(k)
                except:
                    logger.error("Bad line at line: %d (1-based), content: '%s'." % (line_cnt, line))
                    raise

        d.unk_word = '[UNK]'
        d.pad_word = '[PAD]'
        d.eos_word = '[SEP]'
        d.bos_word = '[CLS]'

        d.bos_index = d.add_symbol('[CLS]')
        d.pad_index = d.add_symbol('[PAD]')
        d.eos_index = d.add_symbol('[SEP]')
        d.unk_index = d.add_symbol('[UNK]')

        d.nspecial = 999
        return d

    def save(self, f):
        """Stores dictionary into a text file"""
        ex_keys, ex_vals = self._get_meta()
        self._save(f, zip(ex_keys + self.symbols, ex_vals + self.count))
