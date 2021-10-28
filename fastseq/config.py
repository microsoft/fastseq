# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Fastseq related configurations"""

import os

# fastseq environment variables
FASTSEQ_DEFAULT_LOG_LEVEL = 'INFO'
FASTSEQ_LOG_LEVEL = os.getenv('FASTSEQ_LOG_LEVEL', FASTSEQ_DEFAULT_LOG_LEVEL)
FASTSEQ_CACHE_DIR = os.getenv('FASTSEQ_CACHE_DIR', os.path.join(os.sep, 'tmp'))
FASTSEQ_UNITTEST_LOG_XML_DIR = os.getenv(
    'FASTSEQ_UNITTEST_LOG_XML_DIR', os.path.join('tests', 'log_xml'))

FASTSEQ_LOG_FORMAT = (
    '%(levelname)s %(asctime)s %(pathname)s:%(lineno)d] %(message)s')

FASTSEQ_VERSION = '0.2.0'

# supported versions of transformers
MIN_TRANSFORMERS_VERSION = '4.12.0'
MAX_TRANSFORMER_VERSION = '4.12.0'

# supported versions of fairseq
MIN_FAIRSEQ_VERSION = '0.10.0'
MAX_FAIRSEQ_VERSION = '0.10.2'

#Set following variable to use Efficient-Lossless Attention
USE_EL_ATTN = True if os.getenv('USE_EL_ATTN', '0') == '1' else False
