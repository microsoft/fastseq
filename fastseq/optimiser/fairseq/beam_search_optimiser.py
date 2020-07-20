# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Import the optimization for beam search related parts in FairSeq."""

from absl import logging
import fairseq
from fairseq.sequence_generator import SequenceGenerator

LATEST_VERSION = 'latest'


def get_fairseq_version():
    """Return the version of fairseq as a string.

    If it is installed from the github master branch, return 'latest'.

    Returns:
        A string of fairseq version.
    """

    version = fairseq.__version__
    # TODO: Is there a better way to identify the latest release and the master
    # branch?
    if version == '0.9.0' and hasattr(SequenceGenerator, 'finalize_hypos'):
        return LATEST_VERSION
    return version


def apply_fairseq_optimization():
    """Automaticall apply the optimization to the installed fairseq.

    The optimized classes and functions are replaced in runtime. Currently, only
    `0.9.0` and `latest` versions of fairseq are supported.
    """
    version = get_fairseq_version()
    message = "fairseq == {} has been optimized.".format(version)
    if version == '0.9.0':
        import fastseq.optimiser.fairseq.beam_search_optimiser_v1  # pylint: disable=import-outside-toplevel
        logging.info(message)
        return
    if version == LATEST_VERSION:
        import fastseq.optimiser.fairseq.beam_search_optimiser_v2  # pylint: disable=import-outside-toplevel
        logging.info(message)
        return

    logging.warning(
        "fairseq == {} is not supported yet, please upgrade it to 0.9.0 or above" # pylint: disable=line-too-long
        .format(version))
