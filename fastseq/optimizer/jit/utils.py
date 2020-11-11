# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Utilities for graph rewriting"""

import torch

def rewrite_graph(pattern: str,
                  rewrite_pattern: str,
                  input_graph: torch._C.Graph):
    torch._C._jit_pass_custom_pattern_based_rewrite_graph(
        pattern, rewrite_pattern, input_graph)


def graph_pattern(obj):
    def convert_to_graph_pattern():
      script = torch.jit.script(obj)
      return script.graph.str()

    return convert_to_graph_pattern
