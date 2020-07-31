# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""
Automatically apply the optimizations if the supported versions of FairSeq
are detected.
"""

from absl import logging

try:
    import fairseq
    from fastseq.optimizer.fairseq.beam_search_optimizer import apply_fairseq_optimization
    apply_fairseq_optimization()
except ImportError as error:
    logging.warning(error)
