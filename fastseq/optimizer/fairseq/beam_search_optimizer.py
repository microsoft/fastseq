# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Import the optimization for beam search related parts in FairSeq."""

from packaging import version

import fairseq
from fairseq.models import ARCH_MODEL_REGISTRY, MODEL_REGISTRY
from fairseq.sequence_generator import SequenceGenerator

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
    if v == '0.9.0' and hasattr(SequenceGenerator, 'finalize_hypos'):
        return LATEST_VERSION
    return v


def apply_fairseq_optimization():
    """Automaticall apply the optimization to the installed fairseq.

    The optimized classes and functions are replaced in runtime. Currently, only
    `0.9.0` and `latest` versions of fairseq are supported.
    """

    v = version.parse(get_fairseq_version())

    if v == version.parse('0.9.0'):
        import fastseq.optimizer.fairseq.beam_search_optimizer_v1  # pylint: disable=import-outside-toplevel
        import fastseq.optimizer.fairseq.generate_v1  # pylint: disable=import-outside-toplevel
        _update_fairseq_model_registration()
        logger.debug("fairseq == {} has been optimized.".format(v))
        return

    if v > version.parse('0.9.0') or isinstance(v, version.LegacyVersion):
        import fastseq.optimizer.fairseq.beam_search_optimizer_v2  # pylint: disable=import-outside-toplevel
        import fastseq.optimizer.fairseq.generate_v2  # pylint: disable=import-outside-toplevel
        _update_fairseq_model_registration()
        logger.debug("fairseq == {} has been optimized.".format(v))
        return
    logger.warning(
        "fairseq == {} is not supported yet, please upgrade it to 0.9.0 or above" # pylint: disable=line-too-long
        .format(v))


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
