# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Optmize einsum operation in the graph"""

from typing import List

import torch
from torch import Tensor

from fastseq.optimizer.jit.utils import graph_pattern, rewrite_graph

def einsum_pattern_0(t0: str, t1: List[Tensor]):
    r = torch.einsum(t0, t1)
    return r

def einsum_rewrite_pattern_0(eqn: str, operands: List[Tensor]):
    # eqn = eqn.replace(' ', '')  # TODO: fix the issue: ValueError: stoll
    # for cases like "bmhtd,bnhsd->bmhts"
    if (len(eqn) == 18 and eqn[0:4] == eqn[13:17] and eqn[0] == eqn[6] and
        eqn[2] == eqn[8] and eqn[4] == eqn[10] and eqn[9] == eqn[17]):
        t0 = operands[0]
        t1 = operands[1]
        b, m, h, t, d = t0.shape
        s = t1.size(3)
        n = t1.size(1)
        t1 = t1.permute(0, 2, 3, 4, 1) # (b, h, s, d, n)
        if n > 1:
            t1 = t1.sum(dim=4, keepdim=True) # (b, h, s, d, 1)

        t0 = t0.permute(0, 2, 1, 3, 4) # (b, h, m, t, d)
        t1 = t1.permute(0, 1, 3, 4, 2) # (b, h, d, 1, s)
        t0 = t0.reshape(b*h, m*t, d)
        t1 = t1.view(b*h, d, s)
        r = torch.bmm(t0, t1).view(b, h, m, t, s).permute(0, 2, 1, 3, 4)
        return r

    # for cases like "bmhts,bnhsd->bmhtd"
    if (len(eqn) == 18 and eqn[0:4] == eqn[13:17] and eqn[0] == eqn[6] and
        eqn[2] == eqn[8] and eqn[4] == eqn[9] and eqn[10] == eqn[17]):
        t0 = operands[0]
        t1 = operands[1]
        b, m, h, t, s = t0.shape
        n = t1.size(1)
        d = t1.size(4)
        t1 = t1.permute(0, 2, 4, 3, 1) # (b, h, d, s, n)
        if n > 1:
            t1 = t1.sum(dim=4, keepdim=True) # (b, h, d, s, 1)
        # t1 = t1.squeeze(1) # (b, h, s, d)
        t0 = t0.permute(0, 2, 1, 3, 4) # (b, h, m, t, s)
        t1 = t1.permute(0, 1, 3, 4, 2) # (b, h, s, 1, d)
        t0 = t0.reshape(b*h, m*t, s)
        t1 = t1.view(b*h, s, d)
        r = torch.bmm(t0, t1).view(b, h, m, t, d).permute(0, 2, 1, 3, 4)
        return r

    return torch.einsum(eqn, operands)

EINSUM_PATTERN_STR = graph_pattern(einsum_pattern_0)()
EINSUM_REWRITE_PATTERN_STR = graph_pattern(einsum_rewrite_pattern_0)()

def rewrite_einsum(input_graph: torch._C.Graph):
    rewrite_graph(EINSUM_PATTERN_STR, EINSUM_REWRITE_PATTERN_STR, input_graph)
