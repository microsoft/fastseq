# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""
Automatically apply the optimizations if the supported versions of FairSeq
are detected.
"""

from packaging import version

from fastseq.config import MAX_FAIRSEQ_VERSION, MIN_FAIRSEQ_VERSION
from fastseq.logging import get_logger
from fastseq.utils.api_decorator import FAIRSEQ_OPTIMIZED_CLASSES

logger = get_logger(__name__)

LATEST_VERSION = 'latest'


def get_fairseq_version():
    """Return the version of fairseq as a string.

    If it is installed from the github master branch, return 'latest'.

    Returns:
        A string of fairseq version.
    """

    v = fairseq.__version__
    # TODO: find a better way to identify the latest release and the master
    # branch.
    if hasattr(SequenceGenerator, 'finalize_hypos'):
        return LATEST_VERSION
    return v

def is_supported_fairseq():
    """Check if the installed fairseq is supported.

    Returns:
        a bool value: True indicates the installed fairseq is supported.
    """

    v = get_fairseq_version()
    if v == LATEST_VERSION:
        return False

    v = version.parse(v)
    return version.parse(
        MIN_FAIRSEQ_VERSION) <= v <= version.parse(MAX_FAIRSEQ_VERSION)

def apply_fairseq_optimization():
    """Automaticall apply the optimization to the installed fairseq.

    The optimized classes and functions are replaced in runtime. Currently, only
    `0.9.0` and `latest` versions of fairseq are supported.
    """
    v = get_fairseq_version()
    if not is_supported_fairseq():
        logger.warning(
            f"fairseq == {v} is not supported yet, please change it to "
            f"v{MIN_FAIRSEQ_VERSION} ~ v{MAX_FAIRSEQ_VERSION}")
        return

    import fastseq.optimizer.fairseq.beam_search_optimizer  # pylint: disable=import-outside-toplevel
    import fastseq.optimizer.fairseq.generate  # pylint: disable=import-outside-toplevel
    _update_fairseq_model_registration()
    logger.debug(f"fairseq == {v} has been optimized.")


def _update_fairseq_model_registration():
    """Use the optimized classes to update the registered fairseq models and
    arches.

    Args:
        optimized_classes (list): a list of optimized fairseq classes.
    """
    for model_name, model_class in MODEL_REGISTRY.items():
        for optimized_cls in FAIRSEQ_OPTIMIZED_CLASSES:
            if model_class == optimized_cls.__base__:
                MODEL_REGISTRY[model_name] = optimized_cls
                logger.debug(
                    "Update the register model {} from {} to {}".format(
                        model_name, model_class, optimized_cls))
            elif model_class.__base__ == optimized_cls.__base__:
                logger.debug(
                    "Update the base class of {} from {} to {}".format(
                        model_class, model_class.__base__, optimized_cls))
                MODEL_REGISTRY[model_name].__bases__ = (optimized_cls,)

    for arch_name, model_class in ARCH_MODEL_REGISTRY.items():
        for optimized_cls in FAIRSEQ_OPTIMIZED_CLASSES:
            if model_class in optimized_cls.__bases__:
                ARCH_MODEL_REGISTRY[arch_name] = optimized_cls
                logger.debug(
                    "Update the register model arch {} from {} to {}".format(
                        arch_name, model_class, optimized_cls))
            elif model_class.__base__ == optimized_cls.__base__:
                logger.debug(
                    "Update the base class of {} from {} to {}".format(
                        model_class, model_class.__base__, optimized_cls))
                ARCH_MODEL_REGISTRY[arch_name].__bases__ = (optimized_cls,)


try:
    import fairseq # pylint: disable=ungrouped-imports
    from fairseq.models import ARCH_MODEL_REGISTRY, MODEL_REGISTRY # pylint: disable=ungrouped-imports
    from fairseq.sequence_generator import SequenceGenerator # pylint: disable=ungrouped-imports
    apply_fairseq_optimization()
except ImportError as error:
    logger.warning('fairseq can not be imported. Please ignore this warning if '
                   'you are not using fairseq')
