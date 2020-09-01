# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""
Automatically apply the optimizations if the supported versions of transformers
are detected.
"""

from packaging import version
from fastseq.logging import get_logger

logger = get_logger(__name__)

def get_transformers_version():
    """Return the version of transformers as a string.

    Returns:
        A string of fairseq version.
    """
    return transformers.__version__


def apply_transformers_optimization():
    """Automaticall apply the optimization to the installed transformers.

    The optimized classes and functions are replaced in runtime.
    """

    v = version.parse(get_transformers_version())

    if v >= version.parse('3.0.2'):
        import fastseq.optimizer.transformers.beam_search_optimizer # pylint: disable=import-outside-toplevel
        import fastseq.optimizer.transformers.modeling_t5_optimizer # pylint: disable=import-outside-toplevel

        logger.debug("transformers == {} has been optimized.".format(v))
        return

    if v == version.parse('2.11.0'):
        import fastseq.optimizer.transformers.modeling_bart_optimizer_2_11_0 # pylint: disable=import-outside-toplevel
        logger.debug("transformers == {} has been optimized.".format(v))
        return

    logger.warning(
        "transformers == {} is not supported yet, please upgrade it to 3.0.2 or above" # pylint: disable=line-too-long
        .format(v))

try:
    import transformers
    apply_transformers_optimization()
except ImportError as error:
    logger.warning('transformers can not be imported. Please ignore this '
                   'warning if you are not using transformers')
