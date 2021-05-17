# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
"""Register models"""

import logging
import sys
from fastseq.logging import get_logger
logger = get_logger(__name__, logging.INFO)

import fastseq.models.prophetnet_fs
import fastseq.models.unilm_hf

try:
    import fastseq.models.modeling_auto_hf
except ImportError as error:
    logger.warning(
        'transformers can not be imported.')
except:
    logger.error("Unexpected error: {}".format(sys.exc_info()[0]))
    raise
