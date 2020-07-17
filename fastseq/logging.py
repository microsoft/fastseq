import os

import absl
from absl import logging

FASTSEQ_LOG_LEVEL = 'FASTSEQ_LOG_LEVEL'

logging.get_absl_handler().use_absl_log_file()
absl.flags.FLAGS.mark_as_parsed()


def set_log_level(log_level=None):
    level = os.environ.get(
        FASTSEQ_LOG_LEVEL) if log_level is None else log_level

    if level is not None:
        logging.set_verbosity(level)
        return

    logging.set_verbosity(logging.INFO)
