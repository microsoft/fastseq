# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

""" Unit test for Ngram repeat block cuda op """

import math
import torch
from fastseq.ops.ngram_repeat_block import NGramRepeatBlock
from fastseq.utils.test_utils import TestCaseBase
from absl.testing import absltest, parameterized

class NgramRepeatBlockTest(TestCaseBase):
    """ check to ensure cuda implementation output
        of this op matches with original fairseq
        implememntation.
    """

    def setUp(self):
        super(NgramRepeatBlockTest, self).setUp()

    def apply_no_repeat_ngram(self, tokens,lprobs, bsz,step, beam_size):
        """ Fairseq implementation of blocking
            repeated ngrams
        """
        no_repeat_ngram_size = 3
        banned_list = [[] for bbsz_idx in range(bsz * beam_size)]
        cpu_tokens = tokens.cpu()[:, :step + 1].numpy()
        check_start_pos = step + 2 - no_repeat_ngram_size
        for bbsz_idx in range(bsz * beam_size):
            for i in range(check_start_pos):
                is_banned = True
                for k in range(no_repeat_ngram_size - 1):
                    if cpu_tokens[bbsz_idx, i + k] != cpu_tokens[
                        bbsz_idx, check_start_pos + k]:
                        is_banned = False
                        break
                if is_banned:
                    banned_list[bbsz_idx].append(
                        cpu_tokens[bbsz_idx,
                                   i + no_repeat_ngram_size - 1])

        def calculate_banned_tokens(bbsz_idx):
            """before decoding the next token, prevent decoding
            of ngrams that have already appeared
            """
            banned_tokens_per_sample = [
                (bbsz_idx, t) for t in banned_list[bbsz_idx]
            ]
            return banned_tokens_per_sample

        banned_tokens = []
        if step + 2 - no_repeat_ngram_size >= 0:
            for bbsz_idx in range(bsz * beam_size):
                banned_tokens.extend(calculate_banned_tokens(bbsz_idx))

        if banned_tokens:
            banned_tokens = torch.LongTensor(banned_tokens)
            lprobs.index_put_(
                tuple(banned_tokens.t()),
                lprobs.new_tensor([-math.inf] * len(banned_tokens)))

        return lprobs

    @parameterized.named_parameters({
    'testcase_name': 'Normal',
    'vocab_size': 10,
    'bsz': 256,
    'beam_size': 1,
    'step': 6,
    'ngram_repeat_block_size': 3,
    'sequence_length':2048,
    'pos1':0,
    })
    def test_ngram_repeat_block_kernel(self, bsz, beam_size, vocab_size,
                    step, ngram_repeat_block_size, sequence_length, pos1):

        """ Use random input with repeated ngram to check
            whether corresponding token in vocabulary is blocked (-Inf score)

        Args:
        bsz (int): batch size
        beam_size (int): beam size
        vocab_size (int): vocab size
        step (int): current decoding step
        ngram_repeat_block_size (int): size of ngram
        sequence_length (int): sequence length
        pos1 (int) first position where repeated ngram occurs
                    within a sentence.
        """

        lprobs = torch.zeros(bsz*beam_size, 10).type(torch.FloatTensor)
        repeated_ngram = torch.randint(0,10, (1,2))
        #second place where ngram is repeated
        pos2 = step-ngram_repeat_block_size+2
        #Dummy input with repeated ngram
        inp = torch.cat((torch.randint(0,10, (1,pos1)),
                        repeated_ngram, torch.randint(0,10,
                        (1,pos2-pos1-ngram_repeat_block_size+1)),
                        repeated_ngram, torch.randint(0,10,
                        (1, sequence_length -
                        pos2-ngram_repeat_block_size+1))), 1)
        #inp = torch.Tensor([
        #                [1,2,3,7,5,1,2,0,0,0,0,0,0,0,0,0],
        #                [4,9,8,2,3,9,8,0,0,0,0,0,0,0,0,0],
        #                ]).type(torch.LongTensor)
        tokens=inp.repeat( (bsz,1))
        #CUDA kernel initialization
        rnn = NGramRepeatBlock()
        lprobs = lprobs.cuda()
        tokens = tokens.cuda()
        #Cuda opt implementation
        lprobs = rnn( tokens,lprobs, bsz, step, beam_size,
                    ngram_repeat_block_size)
        #Original implementation
        lprobs_ref = self.apply_no_repeat_ngram(tokens, lprobs, bsz,
                                step, beam_size)
        err_msg = '''
        ngram repeat block kernel implementation output
        doesn't match with output of original implementation
        '''
        assert torch.all(torch.eq(lprobs,lprobs_ref)).cpu().numpy(), err_msg

if __name__ == "__main__":
    absltest.main()
