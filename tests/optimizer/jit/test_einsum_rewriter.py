# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import functools
import logging
import timeit

from absl.testing import absltest, parameterized
import torch
from torch import Tensor

from fastseq.logging import get_logger
from fastseq.optimizer.jit.einsum_rewriter import rewrite_einsum
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
         'shape0': [128, 4, 16, 3, 64],
         'shape1': [128, 2, 16, 64, 7]},
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
        repeat_times = 1000

        r0 = run_einsum(eqn, t0, t1)
        time0 = timeit.Timer(functools.partial(run_einsum, eqn, t0, t1))
        s0 = time0.timeit(repeat_times)

        script_run_einsum = torch.jit.script(run_einsum)
        logger.debug(f"Original graph: \n{script_run_einsum.graph.str()}")
        rewrite_einsum(script_run_einsum.graph)
        logger.debug(f"Optimized graph: \n{script_run_einsum.graph.str()}")
        self.assertTrue('bmm' in script_run_einsum.graph.str())

        r1 = script_run_einsum(eqn, t0, t1)
        time1 = timeit.Timer(
            functools.partial(script_run_einsum, eqn, t0, t1))
        s1 = time1.timeit(repeat_times)

        self.assertTrue(torch.equal(r0, r1))
        logger.info(f"einsum took: {s0}; optimized einsum torchscript took: "
                    f"{s1};")


if __name__ == "__main__":
    absltest.main()
