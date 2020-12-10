# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Load and apply the registered graph rewrite patterns"""

import torch

from fastseq.optimizer.jit.einsum_rewriter import rewrite_einsum

def optimize_graph(input_graph: torch._C.Graph):
    rewrite_einsum(input_graph)
