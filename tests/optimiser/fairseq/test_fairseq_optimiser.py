import os

import torch
from absl import logging
from absl.testing import absltest, parameterized

import fastseq
from fairseq.models.bart.model import BARTModel
from fastseq.utils.file_utils import decompress_file, make_dirs, wget
from fastseq.utils.test_utils import (BART_MODEL_URLS, CACHED_BART_MODEL_DIR,
                                      CACHED_BART_MODEL_PATHS, TestCaseBase)


class FairseqBeamSearchOptimiserTest(TestCaseBase):
    def setUp(self):
        super(FairseqBeamSearchOptimiserTest, self).setUp()
        # TODO: create a dummy model instead of loading a large-size model.
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

        self.source_path = 'tests/optimiser/fairseq/data/cnndm_128.txt'

        # read the expected output.
        self.expected_output_path = 'tests/optimiser/fairseq/data/expected_output.hypo'
        self.expected_output = []
        with open(self.expected_output_path, 'rt',
                  encoding="utf-8") as expected_output_file:
            for line in expected_output_file:
                self.expected_output.append(line.strip())

    @parameterized.named_parameters({
        'testcase_name': 'Normal',
        'beam_size': 4,
        'batch_size': 128,
        'need_attn': False,
        'lenpen': 2.0,
        'max_len_b': 140,
        'min_len': 55,
        'no_repeat_ngram_size': 3
    })
    def testFairseqBeamSearchOptimiser(self, beam_size, batch_size, need_attn,
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
        with open(self.source_path, 'rt', encoding="utf-8") as source:
            slines = []
            torch.cuda.synchronize()
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
            self.assertTrue(len(slines) == 0)

            self.assertEqual(len(output), len(self.expected_output))

            for i in range(len(output)):
                if output[i] != self.expected_output[i]:
                    logging.error("\n{} \n v.s. \n{}".format(
                        output[i], self.expected_output[i]))


if __name__ == "__main__":
    absltest.main()
