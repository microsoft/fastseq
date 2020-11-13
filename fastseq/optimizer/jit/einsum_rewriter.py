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
def einsum_rewrite_pattern_0(eqn: str, operands: List[Tensor]):
    # for cases like "bmhtd,bnhsd->bmhts"
    if (len(eqn) == 18 and eqn[0:3] == eqn[13:16] and eqn[0] == eqn[6] and
        eqn[2] == eqn[8] and eqn[4] == eqn[10] and eqn[3] == eqn[16] and
        eqn[9] == eqn[17]):
        t0 = operands[0]
        t1 = operands[1]
        b = t0.size(0)
        m = t0.size(1)
        h = t0.size(2)
        t = t0.size(3)
        d = t0.size(4)
        n = t1.size(1)
        if n > 1:
            t1 = t1.sum(dim=1, keepdim=True) # (b, 1, h, d, s)
        s = t1.size(3)
        t0 = t0.permute(0, 2, 1, 3, 4) # (b, h, m, t, d)
        t1 = t1.permute(0, 2, 1, 4, 3) # (b, h, 1, d, s)
        t0 = t0.reshape(b*h, m*t, d)
        t1 = t1.reshape(b*h, d, s)
        r = torch.bmm(t0, t1).view(b, h, m, t, s).permute(0, 2, 1, 3, 4)
        return r

    # for cases like "bmhts,bnhsd->bmhtd"
    if (len(eqn) == 18 and eqn[0:3] == eqn[13:16] and eqn[0] == eqn[6] and
        eqn[2] == eqn[8] and eqn[4] == eqn[9] and eqn[3] == eqn[16] and
        eqn[10] == eqn[17]):
        t0 = operands[0]
        t1 = operands[1]
        b = t0.size(0)
        m = t0.size(1)
        h = t0.size(2)
        t = t0.size(3)
        s = t0.size(4)
        n = t1.size(1)
        if n > 1:
            t1 = t1.sum(dim=1, keepdim=True) # (b, 1, h, s, d)
        d = t1.size(4)
        t0 = t0.permute(0, 2, 1, 3, 4) # (b, h, m, t, s)
        t1 = t1.permute(0, 2, 1, 3, 4) # (b, h, 1, s, d)
        t0 = t0.reshape(b*h, m*t, s)
        t1 = t1.reshape(b*h, s, d)
        r = torch.bmm(t0, t1).view(b, h, m, t, d).permute(0, 2, 1, 3, 4)
        return r

    return torch.einsum(eqn, operands)

EINSUM_PATTERN_STR = einsum_pattern_0()
EINSUM_REWRITE_PATTERN_STR = einsum_rewrite_pattern_0()

def rewrite_einsum(input_graph: torch._C.Graph):
    rewrite_graph(EINSUM_PATTERN_STR, EINSUM_REWRITE_PATTERN_STR, input_graph)
