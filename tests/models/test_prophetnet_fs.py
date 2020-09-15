# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""
Test the optimizations on FairSeq to make sure the changes do not affect the
model accuracy.
"""

import os
import time
from urllib.parse import urljoin

import torch
from absl.testing import absltest, parameterized

import fastseq
from fastseq.logging import get_logger
from fastseq.models.prophetnet_fs.ngram_s2s_model import NgramTransformerProphetModel
from fastseq.utils.file_utils import decompress_file, make_dirs, wget
from fastseq.utils.test_utils import (PROPHETNET_MODEL_URLS,
                                      CACHED_PROPHETNET_MODEL_PATHS,
                                      TestCaseBase)

logger = get_logger(__name__)

class ProphetNetModelTest(TestCaseBase):
    """Test ProphetNet

    `prophetnet_large_160G_gigaword_model` is used for benchmarking. If it does
    not exist, it will be downloaded first. As the the model is big, it will
    take a while to download. Once downloaded, it will be cached for future
    usage.
    """

    def setUp(self):
        """set up the test environment"""

        super(ProphetNetModelTest, self).setUp()
        prophetnet_dir = CACHED_PROPHETNET_MODEL_PATHS[
            'prophetnet_large_160G_cnndm']
        prophetnet_url_base = PROPHETNET_MODEL_URLS[
            'prophetnet_large_160G_cnndm']
        if not os.path.exists(prophetnet_dir):
            make_dirs(prophetnet_dir)
            for download_file in ['model.pt', 'dict.src.txt', 'dict.tgt.txt']:
                output_path = os.path.join(prophetnet_dir, download_file)
                with open(output_path, 'xb') as fout:
                    download_url = urljoin(prophetnet_url_base, download_file)
                    wget(download_url, fout)

        self.prophetnet = NgramTransformerProphetModel.from_pretrained(
            prophetnet_dir, checkpoint_file='model.pt')

        self.source_path = 'tests/models/data/cnn_dm_128_bert.txt'

        # read the expected output.
        self.expected_output_path = 'tests/models/data/cnn_dm_128_bert_expected_output.hypo'  # pylint: disable=line-too-long
        self.expected_outputs = []
        with open(self.expected_output_path, 'rt',
                  encoding="utf-8") as expected_output_file:
            for line in expected_output_file:
                self.expected_outputs.append(line.strip())

    @parameterized.named_parameters({
        'testcase_name': 'Normal',
        'beam_size': 5,
        'batch_size': 128,
        'need_attn': False,
        'lenpen': 1.2,
        'max_len_b': 110,
        'min_len': 45,
        'no_repeat_ngram_size': 3
    })
    def test_beam_search_optimizer(self, beam_size, batch_size, need_attn,
                                   lenpen, max_len_b, min_len,
                                   no_repeat_ngram_size):
        """Make sure the changes do not affect the model accuracy.

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
        self.prophetnet.model.make_generation_fast_(
            beamable_mm_beam_size=beam_size, need_attn=need_attn)
        self.prophetnet.cuda()
        self.prophetnet.eval()
        processed_sample_count = 0
        outputs = []
        start = time.time()
        with open(self.source_path, 'rt', encoding="utf-8") as source:
            slines = []
            torch.cuda.synchronize()
            for sline in source:
                slines.append(sline.strip())
                if len(slines) % batch_size != 0:
                  continue
                with torch.no_grad():
                    hypotheses_batch = self.prophetnet.sample(
                        slines,
                        beam=beam_size,
                        lenpen=lenpen,
                        max_len_b=max_len_b,
                        min_len=min_len,
                        no_repeat_ngram_size=no_repeat_ngram_size)
                    hypotheses_batch = [
                        output.strip() for output in hypotheses_batch]
                processed_sample_count += len(slines)
                outputs.extend(hypotheses_batch)
                slines = []
            if slines:
                outputs.extend(self.prophetnet.sample(
                    slines,
                    beam=beam_size,
                    lenpen=lenpen,
                    max_len_b=max_len_b,
                    min_len=min_len,
                    no_repeat_ngram_size=no_repeat_ngram_size))
                processed_sample_count += len(slines)

        torch.cuda.synchronize()
        end = time.time()
        logger.info(
            "Finish the processing of {} samples with the speed {:.2f} "
            "samples/second".format(processed_sample_count,
                                    processed_sample_count / (end - start)))

        self.assertEqual(len(outputs), len(self.expected_outputs))

        for i, output in enumerate(outputs):
            self.assertEqual(output, self.expected_outputs[i])

if __name__ == "__main__":
    absltest.main()
