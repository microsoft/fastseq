""" Wrapper for CUDA extension """
from torch import nn
from torch.autograd import Function
import ngrb_cuda

class NGRBFunction(Function):
    def forward(self, tokens, lprobs, bsz,
        step, beam_size, no_repeat_ngram_size):
        outputs = ngrb_cuda.forward(tokens,
        lprobs, bsz, step, beam_size, no_repeat_ngram_size)
        return outputs

class NGRB(nn.Module):
    def __init__(self):
        super(NGRB, self).__init__()

    def reset_parameters(self):
        pass

    def forward(self, tokens, lprobs, bsz,
        step, beam_size, no_repeat_ngram_size):
        return NGRBFunction.apply(tokens, lprobs,
               bsz, step, beam_size, no_repeat_ngram_size)
