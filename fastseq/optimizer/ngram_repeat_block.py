# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

""" Wrapper for ngram_repeat_block cuda extension """
from torch import nn
from torch.autograd import Function
import ngram_repeat_block_cuda

class NGramRepeatBlockFunction(Function):
    """
    forward inputs to ngram_repeat_block cuda extension
    backward method not needed.

    """
    def forward(self, tokens, lprobs, bsz,
        step, beam_size, no_repeat_ngram_size):
        outputs = ngram_repeat_block_cuda.forward(tokens,
        lprobs, bsz, step, beam_size, no_repeat_ngram_size)
        return outputs

    def backward (*args):
        raise NotImplementedError

class NGramRepeatBlock(nn.Module):
    """ Wrapper class for calling ngram_repeat_block cuda extension """
    def __init__(self):
        super(NGramRepeatBlock, self).__init__()

    def reset_parameters(self):
        pass

    def forward(self, tokens, lprobs, bsz,
        step, beam_size, no_repeat_ngram_size):
        return NGramRepeatBlockFunction.apply(tokens, lprobs,
               bsz, step, beam_size, no_repeat_ngram_size)
