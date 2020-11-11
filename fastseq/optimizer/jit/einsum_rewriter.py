# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Optmize einsum operation in the graph"""

from typing import List

import torch
from torch import Tensor

from fastseq.optimizer.jit.utils import graph_pattern, rewrite_graph

@graph_pattern
def einsum_pattern_0(t0: str, t1: List[Tensor]):
    r = torch.einsum(t0, t1)
    return r

@graph_pattern
def einsum_rewrite_pattern_0(equation: str, operands: List[Tensor]):
    if equation == "bmhtd,bnhsd->bmhts":
        t0 = operands[0]
        t1 = operands[1]
        expand_shape = list(t1.shape)
        expand_shape[1] = t0.size(1)
        result_shape = list(t0.shape)
        result_shape[4] = expand_shape[3]
        t1 = t1.expand(expand_shape).transpose(3, 4).contiguous()
        t1 = t1.view(-1, t1.size(3), t1.size(4))
        t0 = t0.view(-1, t0.size(3), t0.size(4))
        r = torch.bmm(t0, t1).view(result_shape)
        return r

    if equation == "bmhts,bnhsd->bmhtd":
        t0 = operands[0]
        t1 = operands[1]
        expand_shape = list(t1.shape)
        expand_shape[1] = t0.size(1)
        result_shape = list(t0.shape)
        result_shape[4] = expand_shape[4]
        t0 = t0.view(-1, t0.size(3), t0.size(4))
        t1 = t1.expand(expand_shape).contiguous()
        t1 = t1.view(-1, t1.size(3), t1.size(4))
        r = torch.bmm(t0, t1).view(result_shape)
        return r

    return torch.einsum(equation, operands)

EINSUM_PATTERN_STR = einsum_pattern_0()
EINSUM_REWRITE_PATTERN_STR = einsum_rewrite_pattern_0()

def rewrite_einsum(input_graph: torch._C.Graph):
    rewrite_graph(EINSUM_PATTERN_STR, EINSUM_REWRITE_PATTERN_STR, input_graph)
