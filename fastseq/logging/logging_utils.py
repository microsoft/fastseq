# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Logging related module."""

import os
import logging

from logging import _checkLevel

FASTSEQ_LOG_LEVEL = 'FASTSEQ_LOG_LEVEL'
FASTSEQ_LOG_FORMAT = (
    '%(levelname)s %(asctime)s %(filename)s:%(lineno)d] %(message)s')

def set_default_log_level():
    if os.environ[FASTSEQ_LOG_LEVEL] is not None:
        fastseq_log_level = _checkLevel(os.environ[FASTSEQ_LOG_LEVEL])
        logging.basicConfig(level=fastseq_log_level,
                            format=FASTSEQ_LOG_FORMAT)
        return
    logging.basicConfig(level=logging.INFO, format=format)

def get_logger(name=None, level=logging.INFO):
    """
    Return a logger with the specific name, creating it if necessary.

    If no name is specified, return the root logger.

    Args:
        name (str, optional): logger name. Defaults to None.

    Returns:
        Logger : the specified logger.
    """
    logger = logging.getLogger(name)
    logger.setLevel(level)
    if os.environ[FASTSEQ_LOG_LEVEL] is not None:
        fastseq_log_level = _checkLevel(os.environ[FASTSEQ_LOG_LEVEL])
        logger.setLevel(fastseq_log_level)
    return logger

def update_all_log_level(level=logging.INFO):
    """
    Update all the loggers to use the specified level.

    Args:
        level (int/str, optional): the log level. Defaults to logging.INFO.
    """
    loggers = [
        logging.getLogger(name) for name in logging.root.manager.loggerDict]
    for logger in loggers:
        logger.setLevel(level)
