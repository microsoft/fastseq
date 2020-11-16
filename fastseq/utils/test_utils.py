# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Utilities to make it easy to add unit tests"""

from inspect import getframeinfo, stack
import os
from statistics import mean, stdev
import time

from absl import flags
from absl.testing import absltest, parameterized

from fastseq.config import FASTSEQ_CACHE_DIR
from fastseq.logging import get_logger
from fastseq.utils.api_decorator import get_class

logger = get_logger(__name__)

FLAGS = flags.FLAGS

def fastseq_test_main():
    caller = getframeinfo(stack()[1][0])
    suffix = '_' + time.strftime("%Y%m%d%H%M%S") + '.xml'
    xml_log_file = caller.filename.replace(os.sep, '_').replace('.py', suffix)
    xml_log_file = os.path.join('tests', 'fastseq_tests', xml_log_file)
    FLAGS.xml_output_file = xml_log_file
    logger.info(f"Fastseq unit test log output filepath: {xml_log_file}")
    absltest.main()

class TestCaseBase(parameterized.TestCase):
    """Base class used for unittest."""


class BenchmarkBase(TestCaseBase):
    """Base class used for benchmark."""

    pass


def benchmark(repeat_times=3):
    """A decorator used to benchmark a method.

    Args:
        repeat_times (int, optional): repeat times to run the method. Defaults
                                      to 3.

    Returns:
        function: a function to repeatedly run the method and record the
                  execution metrics.
    """

    def decorator(func):
        def timeit(*args, **kwargs):
            exec_times = []
            for _ in range(repeat_times):
                start_time = time.time()
                func(*args, **kwargs)
                end_time = time.time()
                exec_times.append(end_time - start_time)
            cls = get_class(func)
            func_name = "{}.{}".format(cls.__name__,
                                       func.__name__) if cls else func.__name__
            avg_time = mean(exec_times)
            stdev_time = stdev(exec_times) if repeat_times > 1 else 0.0
            logger.info(
                "Benchmarking for {} with {} repeat executions: avg = {} seconds, stdev = {}"  # pylint: disable=line-too-long
                .format(func_name, repeat_times, avg_time, stdev_time))

        return timeit

    return decorator


BART_MODEL_URLS = {}
BART_MODEL_URLS[
    'bart.base'] = 'https://dl.fbaipublicfiles.com/fairseq/models/bart.base.tar.gz'
BART_MODEL_URLS[
    'bart.large'] = 'https://dl.fbaipublicfiles.com/fairseq/models/bart.large.tar.gz'
BART_MODEL_URLS[
    'bart.large.mnli'] = 'https://dl.fbaipublicfiles.com/fairseq/models/bart.large.mnli.tar.gz'
BART_MODEL_URLS[
    'bart.large.cnn'] = 'https://dl.fbaipublicfiles.com/fairseq/models/bart.large.cnn.tar.gz'
BART_MODEL_URLS[
    'bart.large.xsum'] = 'https://dl.fbaipublicfiles.com/fairseq/models/bart.large.xsum.tar.gz'

CACHED_BART_MODEL_DIR = os.path.join(FASTSEQ_CACHE_DIR, 'fairseq_bart_models')

CACHED_BART_MODEL_PATHS = {}
CACHED_BART_MODEL_PATHS['bart.base'] = os.path.join(CACHED_BART_MODEL_DIR,
                                                    'bart.base')
CACHED_BART_MODEL_PATHS['bart.large'] = os.path.join(CACHED_BART_MODEL_DIR,
                                                     'bart.large')
CACHED_BART_MODEL_PATHS['bart.large.mnli'] = os.path.join(
    CACHED_BART_MODEL_DIR, 'bart.large.mnli')
CACHED_BART_MODEL_PATHS['bart.large.cnn'] = os.path.join(
    CACHED_BART_MODEL_DIR, 'bart.large.cnn')
CACHED_BART_MODEL_PATHS['bart.large.xsum'] = os.path.join(
    CACHED_BART_MODEL_DIR, 'bart.large')

PROPHETNET_MODEL_URLS = {}
PROPHETNET_MODEL_URLS[
    'prophetnet_large_160G_cnndm'] = 'https://fastseq.blob.core.windows.net/data/models/prophetnet_large_160G_cnndm_model/'
CACHED_PROPHETNET_DIR = os.path.join(FASTSEQ_CACHE_DIR, 'prophetnet')
CACHED_PROPHETNET_MODEL_PATHS = {}
CACHED_PROPHETNET_MODEL_PATHS[
    'prophetnet_large_160G_cnndm'] = os.path.join(
        CACHED_PROPHETNET_DIR, 'prophetnet_large_160G_cnndm')
