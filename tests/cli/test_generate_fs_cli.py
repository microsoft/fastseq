# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""
Test the optimizations on FairSeq to make sure the changes do not affect the
model accuracy.
"""

import os
import subprocess

import torch
from absl.testing import absltest, parameterized
from fairseq.models.bart.model import BARTModel

from fastseq.logging import get_logger

from fastseq.utils.file_utils import decompress_file, make_dirs, wget
from fastseq.utils.test_utils import (BART_MODEL_URLS, CACHED_BART_MODEL_DIR,
                                      CACHED_BART_MODEL_PATHS,  CNNDM_URL, CACHED_CNNDM_DATA_DIR,
                                      fastseq_test_main, TestCaseBase)

logger = get_logger(__name__)

class FairseqGenerateCLITest(TestCaseBase):
    """Test the optimizations on FairSeq

    `bart.large.cnn` model is used for benchmarking. If it does not exist, it
    will be downloaded first. As the the model is big, it will take a while to
    download. Once downloaded, it will be cached for future usage.
    """

    def setUp(self):
        """set up the test environment"""

        super(FairseqGenerateCLITest, self).setUp()
        # TODO: create a dummy model instead of loading a large-size model.
        if not os.path.exists(CACHED_BART_MODEL_PATHS['bart.large.cnn']):
            make_dirs(CACHED_BART_MODEL_DIR, exist_ok=True)
            tar_model_path = os.path.join(CACHED_BART_MODEL_DIR,
                                          'bart.large.cnn.tar.gz')
            with open(tar_model_path, 'xb') as tar_model_file:
                wget(BART_MODEL_URLS['bart.large.cnn'], tar_model_file)
            decompress_file(tar_model_path, CACHED_BART_MODEL_DIR)

        self.source_path = CACHED_CNNDM_DATA_DIR
        make_dirs(self.source_path, exist_ok=True)
        file_list = ["dict.source.txt", "dict.target.txt", "valid.source-target.source.bin", "valid.source-target.target.bin", "valid.source-target.source.idx", "valid.source-target.target.idx"]
        for f in file_list:
            f_path = os.path.join(self.source_path, f)
            if not os.path.exists(f_path):
                with open(f_path, 'xb') as new_file:
                    wget(os.path.join(CNNDM_URL, f), new_file)
                    new_file.close()
            
        self.bart_path = CACHED_BART_MODEL_PATHS['bart.large.cnn'] + '/model.pt'

    @parameterized.named_parameters({
        'testcase_name': 'Normal',
        'beam_size': 4,
        'batch_size': 16,
        'lenpen': 2.0,
        'max_len_b': 140,
        'min_len': 55,
        'no_repeat_ngram_size': 3,
    })
    def test_generate_cli(self, beam_size, batch_size,
                                   lenpen, max_len_b, min_len,
                                   no_repeat_ngram_size):
        """Test the command line interface for fastseq. Make sure the changes do not 
            affect the model accuracy for beam search optimization and el attn optimization

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
        options = ["--path", self.bart_path,
                    "--task", "translation",
                    "--batch-size", str(batch_size),
                    "--gen-subset", "valid",
                    "--truncate-source",
                    "--bpe", "gpt2",
                    "--beam", str(beam_size),
                    "--num-workers", "4",
                    "--min-len", str(min_len),
                    "--max-len-b", str(max_len_b),
                    "--no-repeat-ngram-size", str(no_repeat_ngram_size),
                    "--lenpen", str(lenpen),
                    "--skip-invalid-size-inputs-valid-test",
                    self.source_path]
        fairseq_outs = subprocess.check_output(['fairseq-generate'] + options).decode("utf-8").split("\n")
        try:
            import fastseq
        except ImportError:
            logger.error("Failed to import fastseq")

        # test beam search opt
        options.append("--decode-hypothesis")
        # debug
        import os
        import distutils.sysconfig
        pre = distutils.sysconfig.get_config_var("prefix")
        bindir = os.path.join(pre, "bin")
        print(bindir)
        # end debug
        fastseq_outs = subprocess.check_output(['fastseq-generate-for-fairseq'] + options).decode("utf-8").split("\n")
        # only compare decoded hypotheses
        fairseq_outs = [l.split() for l in fairseq_outs]
        fairseq_outs = [l for l in fairseq_outs if len(l) > 2 and l[0][0] is 'D']
        fastseq_outs = [l.split() for l in fastseq_outs]
        fastseq_outs = [l for l in fastseq_outs if len(l) > 2 and l[0][0] is 'D']
        assert len(fairseq_outs) == len(fastseq_outs)
        assert len(fairseq_outs) == 128
        for i, expected_out in enumerate(fairseq_outs):
            self.assertEqual(expected_out[2:], fastseq_outs[i][2:])
        
        fastseq_outs = None

        # test el attn opt
        options.append("--use-el-attn")
        try: 
            fastseq_outs = subprocess.check_output(['fastseq-generate-for-fairseq'] + options).decode("utf-8").split("\n")
        except subprocess.CalledProcessError as error:
            print('Error code:', error.returncode, '. Output:', error.output.decode("utf-8"))
        # only compare decoded hypotheses
        fastseq_outs = [l.split() for l in fastseq_outs]
        fastseq_outs = [l for l in fastseq_outs if len(l) > 2 and l[0][0] is 'D']
        assert len(fairseq_outs) == len(fastseq_outs)
        assert len(fairseq_outs) == 128
        for i, expected_out in enumerate(fairseq_outs):
            self.assertEqual(expected_out[2:], fastseq_outs[i][2:])


if __name__ == "__main__":
    fastseq_test_main()
