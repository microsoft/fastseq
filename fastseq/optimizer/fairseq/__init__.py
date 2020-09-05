# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""
Automatically apply the optimizations if the supported versions of FairSeq
are detected.
"""

from fastseq.logging import get_logger

logger = get_logger(__name__)

try:
    import fairseq
    from fastseq.optimizer.fairseq.beam_search_optimizer import apply_fairseq_optimization # pylint: disable=ungrouped-imports
    apply_fairseq_optimization()
except ImportError as error:
    logger.warning('fairseq can not be imported. Please ignore this warning if '
                   'you are not using fairseq')
