# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Benchmark the optimizations on FairSeq"""

import os
import time

import torch
from absl import logging
from absl.testing import absltest, parameterized
from fairseq.models.bart.model import BARTModel

import fastseq
from fastseq.utils.file_utils import decompress_file, make_dirs, wget
from fastseq.utils.test_utils import (BART_MODEL_URLS, CACHED_BART_MODEL_DIR,
                                      CACHED_BART_MODEL_PATHS, BenchmarkBase,
                                      benchmark)


class FairseqBeamSearchOptimizerBenchmark(BenchmarkBase):
    """Benchmark the optimizations on FairSeq

    `bart.large.cnn` model is used for benchmarking. If it does not exist, it
    will be downloaded first. As the the model is big, it will take a while to
    download. Once downloaded, it will be cached for future usage.
    """
    def setUp(self):
        """Set up the test environment.
        """
        super(FairseqBeamSearchOptimizerBenchmark, self).setUp()
        if not os.path.exists(CACHED_BART_MODEL_PATHS['bart.large.cnn']):
            make_dirs(CACHED_BART_MODEL_DIR, exist_ok=True)
            tar_model_path = os.path.join(CACHED_BART_MODEL_DIR,
                                          'bart.large.cnn.tar.gz')
            with open(tar_model_path, 'xb') as tar_model_file:
                wget(BART_MODEL_URLS['bart.large.cnn'], tar_model_file)
            decompress_file(tar_model_path, CACHED_BART_MODEL_DIR)

        self.bart = BARTModel.from_pretrained(
            CACHED_BART_MODEL_PATHS['bart.large.cnn'],
            checkpoint_file='model.pt')
        self.source_path = 'tests/optimizer/fairseq/data/cnndm_128.txt'

    @parameterized.named_parameters(
        {
            'testcase_name': 'BSZ=32',
            'beam_size': 4,
            'batch_size': 32,
            'need_attn': False,
            'lenpen': 2.0,
            'max_len_b': 140,
            'min_len': 55,
            'no_repeat_ngram_size': 3
        }, {
            'testcase_name': 'BSZ=64',
            'beam_size': 4,
            'batch_size': 64,
            'need_attn': False,
            'lenpen': 2.0,
            'max_len_b': 140,
            'min_len': 55,
            'no_repeat_ngram_size': 3
        }, {
            'testcase_name': 'BSZ=128',
            'beam_size': 4,
            'batch_size': 128,
            'need_attn': False,
            'lenpen': 2.0,
            'max_len_b': 140,
            'min_len': 55,
            'no_repeat_ngram_size': 3
        })
    @benchmark(repeat_times=1)
    def test_beam_search_optimizer(self, beam_size, batch_size, need_attn,
                                   lenpen, max_len_b, min_len,
                                   no_repeat_ngram_size):
        """benchmark the performance

        Args:
            beam_size (int): beam size.
            batch_size (int): batch size.
            need_attn (bool): indicate if attention is needed.
            lenpen (float): length penalty, where <1.0 favors shorter, >1.0
                            favors longer sentences.
            max_len_b (int): max length of generated text.
            min_len (int): min length of generated text.
            no_repeat_ngram_size (int): size of no repeat gram.
        """
        self.bart.model.make_generation_fast_(beamable_mm_beam_size=beam_size,
                                              need_attn=need_attn)
        self.bart.cuda()
        self.bart.eval()
        self.bart.half()
        count = 0
        sample_num = (128 / batch_size) * batch_size
        output = []
        with open(self.source_path, 'r+', encoding="utf-8") as source:
            slines = []
            torch.cuda.synchronize()
            start = time.time()
            for sline in source:
                slines.append(sline.strip())
                count += 1
                if count % batch_size == 0:
                    with torch.no_grad():
                        hypotheses_batch = self.bart.sample(
                            slines,
                            beam=beam_size,
                            lenpen=lenpen,
                            max_len_b=max_len_b,
                            min_len=min_len,
                            no_repeat_ngram_size=no_repeat_ngram_size)
                        hypotheses_batch = [
                            output.strip() for output in hypotheses_batch
                        ]
                    output.extend(hypotheses_batch)
                    slines = []

            torch.cuda.synchronize()
            end = time.time()
            run_time = (end - start)
            logging.info(
                'BartModel benchmark: {:.2f} s, {:.1f} sample/s'.format(
                    run_time, sample_num / run_time))
            self.assertTrue(len(slines) == 0)


if __name__ == "__main__":
    absltest.main()
