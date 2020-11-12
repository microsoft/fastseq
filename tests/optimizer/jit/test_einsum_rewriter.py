# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

from typing import List

from absl.testing import absltest
import torch
from torch import Tensor

from fastseq.optimizer.jit.einsum_rewriter import rewrite_einsum
from fastseq.utils.test_utils import TestCaseBase

class EinsumRewriterTest(TestCaseBase):

    def test_einsum_rewriter(self):

        def run_einsum(t0: Tensor, t1: Tensor):
            r = torch.einsum("bmhtd,bnhsd->bmhts", t0, t1)
            r = r + 2.0
            return r

        t0 = torch.randn(10, 3, 4, 3, 9, dtype=torch.float32)
        t1 = torch.randn(10, 1, 4, 7, 9, dtype=torch.float32)

        r0 = run_einsum(t0, t1)

        script_run_einsum = torch.jit.script(run_einsum)
        rewrite_einsum(script_run_einsum.graph)
        r1 = script_run_einsum(t0, t1)

        self.assertTrue(torch.equal(r0, r1))

if __name__ == "__main__":
    absltest.main()
