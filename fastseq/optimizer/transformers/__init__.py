# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""
Automatically apply the optimizations if the supported versions of transformers
are detected.
"""

from packaging import version

from fastseq.config import MIN_TRANSFORMERS_VERSION, MAX_TRANSFORMER_VERSION
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
    is_supported_version = version.parse(
        MIN_TRANSFORMERS_VERSION) <= v <= version.parse(MAX_TRANSFORMER_VERSION)

    if not is_supported_version:
        logger.warning(
            f"transformers == {v} is not supported yet, please change it to "
            f"v{MIN_TRANSFORMERS_VERSION} to v{MAX_TRANSFORMER_VERSION}, or try"
            f" other versions of fastseq.")
        return

    import fastseq.optimizer.transformers.modeling_bart_optimizer # pylint: disable=import-outside-toplevel
    import fastseq.optimizer.transformers.modeling_t5_optimizer # pylint: disable=import-outside-toplevel
    import fastseq.optimizer.transformers.beam_search_optimizer # pylint: disable=import-outside-toplevel

    logger.debug(f"transformers == {v} has been optimized.")


try:
    import transformers
    apply_transformers_optimization()
except ImportError as error:
    logger.warning('transformers can not be imported. Please ignore this '
                   'warning if you are not using transformers')
