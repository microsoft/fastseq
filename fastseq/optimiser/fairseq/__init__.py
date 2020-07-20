# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

from absl import logging

# Automatically optimize fairseq if it has been installed.
try:
    import fairseq
    from fastseq.optimiser.fairseq.beam_search_optimiser import apply_fairseq_optimization
    apply_fairseq_optimization()
except ImportError as error:
    logging.warning(error)
