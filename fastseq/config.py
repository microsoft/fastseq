# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Fastseq related configurations"""

import os

# fastseq environment variables
FASTSEQ_DEFAULT_LOG_LEVEL = 'INFO'
FASTSEQ_LOG_LEVEL = os.getenv('FASTSEQ_LOG_LEVEL', FASTSEQ_DEFAULT_LOG_LEVEL)
FASTSEQ_CACHE_DIR = os.getenv('FASTSEQ_CACHE_DIR', os.path.join(os.sep, 'tmp'))

FASTSEQ_LOG_FORMAT = (
    '%(levelname)s %(asctime)s %(filename)s:%(lineno)d] %(message)s')

# supported versions of transformers
MIN_TRANSFORMERS_VERSION = '3.0.2'
MAX_TRANSFORMER_VERSION = '3.0.2'

# supported versions of fairseq
MIN_FAIRSEQ_VERSION = '0.9.0'
MAX_FAIRSEQ_VERSION = '0.9.0'
