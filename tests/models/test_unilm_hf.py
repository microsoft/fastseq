# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""
Test the optimizations on FastSeq to make sure the changes do not affect the
model accuracy.
"""

import os
import time
from urllib.parse import urljoin

import torch
from absl.testing import absltest, parameterized

import fastseq
from fastseq.logging import get_logger
from fastseq.models.unilm_hf.modeling_unilm import UnilmForSeq2Seq
from fastseq.models.unilm_hf.tokenization_unilm import UnilmTokenizer
from fastseq.utils.test_utils import fastseq_test_main, TestCaseBase

logger = get_logger(__name__)

class UnilmModelTest(TestCaseBase):
    """Test Unilm

    `cnndm-unilm-base-cased` is used for benchmarking. If it does
    not exist, it will be downloaded first. As the the model is big, it will
    take a while to download. Once downloaded, it will be cached for future
    usage.
    """

    def setUp(self):
        """set up the test environment"""

        super().setUp()
        self.unilm_model = UnilmForSeq2Seq.from_pretrained('cnndm-unilm-base-cased')
        self.unilm_tokenizer = UnilmTokenizer.from_pretrained('cnndm-unilm-base-cased')
        self.unilm_tokenizer.model_max_length = 608

        self.source_path = 'tests/models/data/cnn_dm_128_bert.txt'

        # read the expected output.
        self.expected_output_path = 'tests/models/data/expected_unilm_base_cased_output.hypo'  # pylint: disable=line-too-long
        self.expected_outputs = []
        with open(self.expected_output_path, 'rt',
                  encoding="utf-8") as expected_output_file:
            for line in expected_output_file:
                self.expected_outputs.append(line.strip())

    @parameterized.named_parameters({
        'testcase_name': 'Normal',
        'beam_size': 5,
        'batch_size': 32,
        'lenpen': 1.0,
        'max_len_b': 160,
        'no_repeat_ngram_size': 3
    })
    def test_beam_search_optimizer(self, beam_size, batch_size,
                                   lenpen, max_len_b=48, min_len=0,
                                   no_repeat_ngram_size=0):
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
        self.unilm_model.cuda()
        self.unilm_model.eval()
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
                    slines_batch = self.unilm_tokenizer.batch_encode_plus(slines, padding="max_length", truncation=True)
                    slines_input_ids = torch.tensor(slines_batch["input_ids"]).cuda()
                    slines_attention_mask = torch.tensor(slines_batch["attention_mask"]).cuda()
                    hypotheses_batch = self.unilm_model.generate(
                        input_ids=slines_input_ids,
                        attention_mask=slines_attention_mask,
                        decoder_start_token_id=None,
                        num_beams=beam_size,
                        length_penalty=lenpen,
                        max_length=max_len_b,
                        min_gen_length=min_len,
                        no_repeat_ngram_size=no_repeat_ngram_size
                    )
                    hypotheses_batch = self.unilm_tokenizer.batch_decode(hypotheses_batch, skip_special_tokens=True, clean_up_tokenization_spaces=False)
                    hypotheses_batch = [
                        output.strip() for output in hypotheses_batch]
                processed_sample_count += len(slines)
                outputs.extend(hypotheses_batch)
                slines = []
            if slines:
                slines_batch = self.unilm_tokenizer.batch_encode_plus(slines, padding="max_length", truncation=True)
                slines_input_ids = torch.tensor(slines_batch["input_ids"]).cuda()
                slines_attention_mask = torch.tensor(slines_batch["attention_mask"]).cuda()
                hypotheses_batch = self.unilm_model.generate(
                    input_ids=slines_input_ids,
                    attention_mask=slines_attention_mask,
                    decoder_start_token_id=None,
                    num_beams=beam_size,
                    length_penalty=lenpen,
                    max_length=max_len_b,
                    min_gen_length=min_len,
                    no_repeat_ngram_size=no_repeat_ngram_size
                )
                hypotheses_batch = self.unilm_tokenizer.batch_decode(hypotheses_batch, skip_special_tokens=True, clean_up_tokenization_spaces=False)
                hypotheses_batch = [
                    output.strip() for output in hypotheses_batch]
                outputs.extend(hypotheses_batch)
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
    fastseq_test_main()
