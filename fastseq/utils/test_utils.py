import os
from statistics import mean, stdev
import time

from absl import logging
from absl.testing import parameterized

from fastseq.utils.api_decorator import get_class


class TestCaseBase(parameterized.TestCase):
    def tearDown(self):
        print('Log output path: {}'.format(logging.get_log_file_name()))


class BenchmarkBase(TestCaseBase):
    pass


def benchmark(repeat_times=3):
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
            logging.info(
                "Benchmarking for {} with {} repeat executions: avg = {} seconds, stdev = {}"
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

CACHED_BART_MODEL_DIR = os.path.join(os.sep, 'tmp', 'fairseq_bart_models')

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
