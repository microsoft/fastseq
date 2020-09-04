# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
""" script for importing fairseq tests """

import glob
import sys
import os
import argparse
import logging
import shutil
import unittest
from git import Repo
from absl.testing import absltest, parameterized

FASTSEQ_PATH = '/'.join(os.path.realpath(__file__).split('/')[0:-2])
FAIRSEQ_PATH = '/tmp/fairseq/'
FAIRSEQ_GIT_URL = 'https://github.com/pytorch/fairseq.git'


class FairseqUnitTests(parameterized.TestCase):
    def prepare_env(self):
        """set env variables"""
        #Removing following path since it contains utils directory
        #which clashes with utils.py file in fairseq/tests.
        if FASTSEQ_PATH in sys.path:
            sys.path.remove(FASTSEQ_PATH)
        sys.path.insert(0, FAIRSEQ_PATH)

    def clone_and_build_fairseq(self, repo, version):
        """clone and build fairseq repo"""
        if os.path.isdir(FAIRSEQ_PATH):
            shutil.rmtree(FAIRSEQ_PATH)
        Repo.clone_from(FAIRSEQ_GIT_URL, FAIRSEQ_PATH, branch=version)
        os.system('pip install git+https://github.com/pytorch/fairseq.git@' +
                  version)
        original_pythonpath = os.environ[
            'PYTHONPATH'] if 'PYTHONPATH' in os.environ else ''
        os.environ['PYTHONPATH'] = FAIRSEQ_PATH + ':' + original_pythonpath

    def get_test_suites(self, test_files_path, blocked_tests):
        """prepare test suite"""
        test_files = [os.path.basename(x) for x in glob.glob(test_files_path)]
        for test in blocked_tests:
            logging.debug('\n .... skipping ....' + test + '\n')
        module_strings = [
            'tests.' + test_file[0:-3] for test_file in test_files
            if test_file not in blocked_tests
        ]
        suites = [
            unittest.defaultTestLoader.loadTestsFromName(test_file)
            for test_file in module_strings
        ]
        return suites

    @parameterized.named_parameters({
        'testcase_name':
        'Normal',
        'without_fastseq_opt':
        False,
        'fairseq_version':
        'v0.9.0',
        'blocked_tests':
        ['test_binaries.py', 'test_bmuf.py', 'test_reproducibility.py']
    })
    def test_suites(self, without_fastseq_opt, fairseq_version, blocked_tests):
        self.clone_and_build_fairseq(FAIRSEQ_GIT_URL, fairseq_version)
        if not without_fastseq_opt:
            import fastseq  #pylint: disable=import-outside-toplevel
        self.prepare_env()
        test_files_path = FAIRSEQ_PATH + '/tests/test_*.py'
        print(test_files_path)
        suites = self.get_test_suites(test_files_path, blocked_tests)
        test_suite = unittest.TestSuite(suites)
        test_runner = unittest.TextTestRunner().run(test_suite)


if __name__ == "__main__":
    absltest.main()
