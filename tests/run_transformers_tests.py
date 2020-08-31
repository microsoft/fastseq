# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

""" script for importing transformers tests """

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
TRANSFORMERS_PATH = '/tmp/transformers/'
TRANSFORMERS_GIT_URL = 'https://github.com/huggingface/transformers.git'


class TransformersUnitTests(parameterized.TestCase):
    def prepare_env(self):
        """set env variables"""
        #Removing following path since it contains utils directory
        #which clashes with utils.py file in transformers/tests.
        if FASTSEQ_PATH in sys.path:
            sys.path.remove(FASTSEQ_PATH)
        sys.path.insert(0, TRANSFORMERS_PATH)
    
    
    def clone_and_build_transformers(self, repo, version):
        """clone and build transformers repo"""
        if os.path.isdir(TRANSFORMERS_PATH):
            shutil.rmtree(TRANSFORMERS_PATH)
        Repo.clone_from(TRANSFORMERS_GIT_URL, TRANSFORMERS_PATH, branch=version)
        os.chdir(TRANSFORMERS_PATH)
        os.system('pip install --editable .')
        original_pythonpath = os.environ['PYTHONPATH'] if 'PYTHONPATH' in os.environ else '' 
        os.environ['PYTHONPATH'] = TRANSFORMERS_PATH + ':' + original_pythonpath
    
    
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
            'testcase_name': 'Normal',
            'without_fastseq_opt': False,
            'transformers_version':'v3.0.2',
            'blocked_tests':[
                            ]
        })
    def test_suites(self, without_fastseq_opt, transformers_version, blocked_tests):
    
        self.clone_and_build_transformers(TRANSFORMERS_GIT_URL, transformers_version)
        if not without_fastseq_opt:
            import fastseq  #pylint: disable=import-outside-toplevel
    
        self.prepare_env()
        test_files_path = TRANSFORMERS_PATH + '/tests/test_*.py'
        suites = self.get_test_suites(test_files_path, blocked_tests)
        test_suite = unittest.TestSuite(suites)
        test_runner = unittest.TextTestRunner().run(test_suite)


if __name__ == "__main__":
    absltest.main()
