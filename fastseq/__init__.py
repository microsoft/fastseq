# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Import core names of FastSeq"""

from fastseq.logging import set_default_log_level
set_default_log_level()

import fastseq.optimizer  # pylint: disable=wrong-import-position
