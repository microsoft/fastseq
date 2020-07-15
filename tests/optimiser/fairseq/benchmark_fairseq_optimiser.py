import time
from absl import logging
from absl.testing import parameterized

import torch

import fastseq

from absl.testing import absltest
from fairseq.models.bart.model import BARTModel
from fastseq.utils.test_util import BenchmarkBase, benchmark


class FairseqBeamSearchOptimiserBenchmark(BenchmarkBase):
    def setUp(self):
        super(FairseqBeamSearchOptimiserBenchmark, self).setUp()
        self.bart = BARTModel.from_pretrained('models/bart.large.cnn/',
                                              checkpoint_file='model.pt')
        self.source_path = 'tests/optimiser/fairseq/data/cnndm_128.txt'

    @parameterized.named_parameters(
        {
            'testcase_name': 'Normal',
            'beam_size': 4,
            'batch_size': 32,
            'need_attn': False,
            'lenpen': 2.0,
            'max_len_b': 140,
            'min_len': 55,
            'no_repeat_ngram_size': 3
        }, {
            'testcase_name': 'LargeBatchSize',
            'beam_size': 4,
            'batch_size': 128,
            'need_attn': False,
            'lenpen': 2.0,
            'max_len_b': 140,
            'min_len': 55,
            'no_repeat_ngram_size': 3
        })
    @benchmark(repeat_times=1)
    def test_beam_search_optimiser(self, beam_size, batch_size, need_attn,
                                   lenpen, max_len_b, min_len,
                                   no_repeat_ngram_size):
        self.bart.model.make_generation_fast_(beamable_mm_beam_size=beam_size,
                                              need_attn=need_attn)
        self.bart.cuda()
        self.bart.eval()
        self.bart.half()
        count = 0
        sample_num = (128 / batch_size) * batch_size
        output = []
        with open(self.source_path) as source:
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
