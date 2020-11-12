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
        b = t0.size(0)
        m = t0.size(1)
        h = t0.size(2)
        t = t0.size(3)
        d = t0.size(4)
        n = t1.size(1)
        s = t1.size(3)
        t0 = t0.permute(0, 2, 1, 3, 4) # (b, h, m, t, d)
        t1 = t1.permute(0, 2, 4, 1, 3) # (b, h, d, n, s)
        t0 = t0.reshape(b*h, m*t, d)
        t1 = t1.reshape(b*h, d, n*s) # TODO: add a check: assert n == 1
        r = torch.bmm(t0, t1).view(b, h, m, t, n*s).permute(0, 2, 1, 3, 4)
        return r

    if equation == "bmhts,bnhsd->bmhtd":
        t0 = operands[0]
        t1 = operands[1]
        b = t0.size(0)
        m = t0.size(1)
        h = t0.size(2)
        t = t0.size(3)
        s = t0.size(4)
        n = t1.size(1)
        d = t1.size(4)
        t0 = t0.permute(0, 2, 1, 3, 4) # (b, h, m, t, s)
        t1 = t1.permute(0, 2, 3, 1, 4) # (b, h, s, n, d)
        t0 = t0.reshape(b*h, m*t, s)
        t1 = t1.reshape(b*h, s, n*d) # TODO: add a check: assert n == 1
        r = torch.bmm(t0, t1).view(b, h, m, t, n*d).permute(0, 2, 1, 3, 4)
        return r

    return torch.einsum(equation, operands)

EINSUM_PATTERN_STR = einsum_pattern_0()
EINSUM_REWRITE_PATTERN_STR = einsum_rewrite_pattern_0()

def rewrite_einsum(input_graph: torch._C.Graph):
    rewrite_graph(EINSUM_PATTERN_STR, EINSUM_REWRITE_PATTERN_STR, input_graph)
