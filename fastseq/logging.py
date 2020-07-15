import os

from absl import logging


def set_log_level():
    LOG_LEVEL = os.environ.get('LOG_LEVEL')

    if LOG_LEVEL is not None:
        logging.set_verbosity(LOG_LEVEL)
    else:
        logging.set_verbosity('info')
