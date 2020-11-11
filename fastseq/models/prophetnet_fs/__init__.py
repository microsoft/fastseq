# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Import ProphetNet modules"""

import logging

from fastseq.logging import get_logger

logger = get_logger(__name__, logging.INFO)

try:
    import fairseq # pylint: disable=ungrouped-imports

    from fastseq.models.prophetnet_fs import translation # pylint: disable=ungrouped-imports
    from fastseq.models.prophetnet_fs import ngram_s2s_model # pylint: disable=ungrouped-imports
    from fastseq.models.prophetnet_fs import ngram_criterions # pylint: disable=ungrouped-imports
except ImportError as error:
    logger.warning(
      'fairseq can not be imported when registering ProphetNet model.')
except:
    logger.error("Unexpected error: {}".format(sys.exc_info()[0]))
    raise
