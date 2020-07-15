from statistics import mean, stdev
import statistics
import time

from absl import logging
from absl.testing import parameterized

from fastseq.utils.api_decorator import get_class


class TestCaseBase(parameterized.TestCase):
    pass


class BenchmarkBase(parameterized.TestCase):
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
