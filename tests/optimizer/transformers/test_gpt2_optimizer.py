# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""
Test the optimizations on Huggingface to make sure the changes do not affect the
model accuracy.
"""
import time

import torch
from absl.testing import parameterized

import fastseq
from fastseq.logging import get_logger
from fastseq.utils.test_utils import fastseq_test_main, TestCaseBase
from transformers import (GPT2LMHeadModel, GPT2Tokenizer)

logger = get_logger(__name__)

class GPT2OptimizerTest(TestCaseBase):
    """Test the optimizations on HuggingFace-transformers-GPT2.
    """
    def setUp(self):
        """Load model, tokenizer and expected output."""

        self.tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
        self.tokenizer.pad_token = self.tokenizer.eos_token
        self.model = GPT2LMHeadModel.from_pretrained('gpt2')

        self.source_path = 'tests/optimizer/transformers/data/cnndm_128.txt'

        # The expected output is generated based on transformers-v4.12.0 with
        # batch_size = 16.
        self.expected_output_path = 'tests/optimizer/transformers/data/expected_gpt2_output.hypo'  # pylint: disable=line-too-long
        self.expected_outputs = []
        with open(self.expected_output_path, 'rt',
                  encoding="utf-8") as expected_output_file:
            for line in expected_output_file:
                self.expected_outputs.append(line.strip())
        self.batch_count = 0

    def _generate(self,
                  slines,
                  max_token_length,
                  num_beams,
                  min_gen_length,
                  max_gen_length,
                  no_repeat_ngram_size,
                  early_stopping,
                  use_cache):
        """Generate the summaries.

        Args:
            slines (List(str)): a list of input sentences.
            max_token_length (int): max tokenized sentence length.
            num_beams (int): beam number.
            min_gen_length (int): min generation length.
            max_gen_length (int): maxium length for the generation output.
            no_repeat_ngram_size (int): size of no repeat gram.
            early_stopping (bool): indicate if the beam search will be early
                stopped.
            use_cache (bool): If `use_cache` is True, past key values are used
                to speed up decoding if applicable to model.

        Returns:
            List(str): a list of generated summaries.
        """
        logger.info("Start to process batch-{}".format(self.batch_count))
        start = time.time()
        with torch.no_grad():
            inputs = self.tokenizer(slines,
                                    max_length=max_token_length,
                                    padding=True,
                                    truncation=True,
                                    return_tensors='pt')

            # Generate Summary
            summary_ids = self.model.generate(
                inputs['input_ids'].cuda(),
                num_beams=num_beams,
                min_length=min_gen_length,
                max_length=max_gen_length,
                no_repeat_ngram_size=no_repeat_ngram_size,
                early_stopping=early_stopping,
                use_cache=use_cache)
            outputs = [self.tokenizer.decode(g) for g in summary_ids]
            self.batch_count += 1
        end = time.time()
        logger.info("Process {} samples in {:.2f} seconds".format(
            len(slines), end - start))
        return outputs

    @parameterized.named_parameters({
        'testcase_name': 'FP32_With_Cache',
        'batch_size': 8,
        'max_token_length': 256,
        'num_beams': 4,
        'min_gen_length': 256 + 50,
        'max_gen_length': 256 + 190,
        'no_repeat_ngram_size': 3,
        'early_stopping': True,
        'use_cache': True,
    })
    def test_beam_search_optimizer(self,
                                   batch_size,
                                   max_token_length,
                                   num_beams,
                                   min_gen_length,
                                   max_gen_length,
                                   no_repeat_ngram_size,
                                   early_stopping,
                                   use_cache):
        """Make sure the changes do not affect the model accuracy.

        Args:
            batch_size (int, optional): batch size. Defaults to 16.
            max_token_length (int, optional): max tokenized sentence length.
                                              Defaults to 1024.
            num_beams (int, optional): beam number. Defaults to 4.
            min_gen_length (int, optional): min generation length. Defaults to
                                            55.
            max_gen_length (int, optional): maxium length for the generation
                                            output. Defaults to 199.
            no_repeat_ngram_size (int, optional): size of no repeat gram.
            early_stopping (bool, optional): indicate if the beam search will be
                                             early stopped.
        """
        self.model.cuda()
        self.model.eval()
        processed_sample_count = 0
        outputs = []
        slines = []
        start = time.time()
        with open(self.source_path, 'rt', encoding="utf-8") as source:
            for sline in source:
                slines.append(sline)
                if len(slines) % batch_size:
                    continue
                outputs.extend(self._generate(
                    slines,
                    max_token_length,
                    num_beams,
                    min_gen_length,
                    max_gen_length,
                    no_repeat_ngram_size,
                    early_stopping,
                    use_cache))
                processed_sample_count += len(slines)
                slines = []

            if slines:
                outputs.extend(self._generate(
                    slines,
                    max_token_length,
                    num_beams,
                    min_gen_length,
                    max_gen_length,
                    no_repeat_ngram_size,
                    early_stopping,
                    use_cache))
                processed_sample_count += len(slines)

            end = time.time()
            logger.info(
                "Finish the processing of {} samples with the speed {:.2f} "
                "samples/second".format(
                    processed_sample_count,
                    processed_sample_count / (end - start)))

        # with open(self.expected_output_path, 'wt',
        #           encoding="utf-8") as expected_output_file:
        #     expected_output_file.write('\n'.join([output.replace('\n', ' ') for output in outputs]))
        for i, output in enumerate(outputs):
            self.assertEqual(output.replace('\n', ' ').strip(), self.expected_outputs[i].strip())

if __name__ == "__main__":
    fastseq_test_main()
