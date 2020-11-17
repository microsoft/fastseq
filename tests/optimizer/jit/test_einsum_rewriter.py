# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import logging
import time

from absl.testing import absltest, parameterized
import torch
from torch import Tensor

from fastseq.logging import get_logger
from fastseq.optimizer.jit.einsum_rewriter import rewrite_einsum, einsum_rewrite_pattern_0
from fastseq.utils.test_utils import TestCaseBase

logger = get_logger(__name__, logging.INFO)

class EinsumRewriterTest(TestCaseBase):

    @parameterized.parameters(
        {'eqn': "bmhtd,bnhsd->bmhts",
         'shape0': [128, 4, 16, 5, 64],
         'shape1': [128, 2, 16, 1024, 64]},
        {'eqn': "kmijd,knisd->kmijs",
         'shape0': [128, 4, 16, 1, 64],
         'shape1': [128, 2, 16, 1024, 64]},
        {'eqn': "bmhts,bnhsd->bmhtd",
         'shape0': [128, 4, 16, 5, 64],
         'shape1': [128, 2, 16, 64, 1024]},
        {'eqn': "impts,inpsw->imptw",
         'shape0': [128, 4, 16, 3, 64],
         'shape1': [128, 2, 16, 64, 7]},
    )
    def test_einsum_rewriter(self, eqn, shape0, shape1):

        def run_einsum(eqn: str, t0: Tensor, t1: Tensor):
            r = torch.einsum(eqn, t0, t1)
            return r

        t0 = torch.randn(shape0, dtype=torch.float32).cuda()
        t1 = torch.randn(shape1, dtype=torch.float32).cuda()
        repeat_times = 1024

        r0 = run_einsum(eqn, t0, t1)
        torch.cuda.synchronize()
        start0 = time.time()
        for _ in range(repeat_times):
            run_einsum(eqn, t0, t1)
        torch.cuda.synchronize()
        end0 = time.time()

        script_run_einsum = torch.jit.script(run_einsum)
        logger.debug(f"Original graph: \n{script_run_einsum.graph.str()}")
        rewrite_einsum(script_run_einsum.graph)
        logger.debug(f"Optimized graph: \n{script_run_einsum.graph.str()}")
        self.assertTrue('bmm' in script_run_einsum.graph.str())

        r1 = script_run_einsum(eqn, t0, t1)
        torch.cuda.synchronize()
        start1 = time.time()
        for _ in range(repeat_times):
            script_run_einsum(eqn, t0, t1)
        torch.cuda.synchronize()
        end1 = time.time()

        r2 = einsum_rewrite_pattern_0(eqn, [t0, t1])
        torch.cuda.synchronize()
        start2 = time.time()
        for _ in range(repeat_times):
            einsum_rewrite_pattern_0(eqn, [t0, t1])
        torch.cuda.synchronize()
        end2 = time.time()

        self.assertTrue(torch.equal(r0, r1))
        self.assertTrue(torch.equal(r0, r2))
        self.assertEqual(
            r0.is_contiguous(), r1.is_contiguous(), r2.is_contiguous())
        logger.info(f"einsum took: {end0 - start0};"
                    f"optimized einsum torchscript took: {end1 - start1};"
                    f"optimized einsum python took: {end2 - start2};")


if __name__ == "__main__":
    absltest.main()
