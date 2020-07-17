import os

import absl
from absl import logging

FASTSEQ_LOG_LEVEL = 'FASTSEQ_LOG_LEVEL'

logging.get_absl_handler().use_absl_log_file()
absl.flags.FLAGS.mark_as_parsed()


def set_log_level(log_level=None):
    """Set the log level.
    
    If there is no log level specified, it will be default to `INFO`.
    
    Args:
        log_level (int/string, optional): the log level. Defaults to None.
    """

    level = os.environ.get(
        FASTSEQ_LOG_LEVEL) if log_level is None else log_level

    if level is not None:
        logging.set_verbosity(level)
        return

    logging.set_verbosity(logging.INFO)
