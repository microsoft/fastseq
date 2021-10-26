# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
""" script for importing fairseq tests """

from fastseq.config import USE_EL_ATTN
import glob
import io
import logging
import os
import shutil
import sys
import time
import unittest

import xmlrunner
from absl.testing import parameterized
from git import Repo
from pip._internal import main as pipmain
from xmlrunner.extra.xunit_plugin import transform

FASTSEQ_PATH = os.sep.join(os.path.realpath(__file__).split('/')[0:-2])
FAIRSEQ_PATH = '/tmp/fairseq/'
FAIRSEQ_GIT_URL = 'https://github.com/pytorch/fairseq.git'

class FairseqUnitTests(parameterized.TestCase):
    """Run all the unit tests under fairseq"""
    def prepare_env(self):
        """set env variables"""
        #Removing following path since it contains utils directory
        #which clashes with utils.py file in fairseq/tests.
        if FASTSEQ_PATH in sys.path:
            sys.path.remove(FASTSEQ_PATH)
        sys.path.insert(0, FAIRSEQ_PATH)
        sys.path.insert(0, '/tmp/')
        print("PATH (os.environ): " + os.environ['PATH'])
        print("PATH (sys.path): " + ' '.join(sys.path))

    def clone_and_build_fairseq(self, repo, version):
        """clone and build fairseq repo"""
        if os.path.isdir(FAIRSEQ_PATH):
            shutil.rmtree(FAIRSEQ_PATH)
        Repo.clone_from(FAIRSEQ_GIT_URL, FAIRSEQ_PATH, branch=version)
        pipmain(['install', '--editable', 'git+https://github.com/pytorch/fairseq.git@' +
                  version + '#egg=fairseq'])
        # print("FAIRSEQ_PATH: " + FAIRSEQ_PATH)
        # directory_path = os.getcwd()
        # print("My current directory is : " + directory_path)
        # print("PATH: " + os.environ['PATH'])
        contents = os.listdir(FAIRSEQ_PATH)
        original_pythonpath = os.environ[
            'PYTHONPATH'] if 'PYTHONPATH' in os.environ else ''
        os.environ['PYTHONPATH'] = FAIRSEQ_PATH + ':' + original_pythonpath
        original_path = os.environ['PATH'] if 'PATH' in os.environ else ''
        os.environ['PATH'] = FAIRSEQ_PATH + ':/tmp/:' + original_path
        

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

    @parameterized.named_parameters(
    {
        'testcase_name': 'Normal',
        'without_fastseq_opt': False,
        'fairseq_version': 'v0.10.2',
        'blocked_tests': [
           'test_binaries.py', 'test_bmuf.py', 'test_reproducibility.py', 
           'test_sequence_generator.py', 'test_backtranslation_dataset.py']
    })
    def test_suites(self, without_fastseq_opt, fairseq_version, blocked_tests):
        """"run test suites"""
        self.clone_and_build_fairseq(FAIRSEQ_GIT_URL, fairseq_version)
        if not without_fastseq_opt:
            import fastseq  # pylint: disable=import-outside-toplevel
        self.prepare_env()
        test_files_path = FAIRSEQ_PATH + '/tests/test_*.py'
        suites = self.get_test_suites(test_files_path, blocked_tests)
        test_suite = unittest.TestSuite(suites)
        test_runner = unittest.TextTestRunner()
        test_result = test_runner.run(test_suite)
        assert len(test_result.errors) == 0

if __name__ == "__main__":
    log_xml_dir = os.getenv(
        'FASTSEQ_UNITTEST_LOG_XML_DIR',
        os.path.join(os.getcwd(), 'tests', 'log_xml'))
    os.makedirs(log_xml_dir, exist_ok=True)
    suffix = '_' + time.strftime("%Y%m%d%H%M%S") + '.xml'
    log_xml_file = __file__.replace(os.sep, '_').replace('.py', suffix)
    log_xml_file = os.path.join(log_xml_dir, log_xml_file)

    out = io.BytesIO()
    unittest.main(
        testRunner=xmlrunner.XMLTestRunner(output=out),
        failfast=False, buffer=False, catchbreak=False, exit=False)
    with open(log_xml_file, 'wb') as report:
        report.write(transform(out.getvalue()))
        print("Save the log of fairseq unit tests into %s" % (log_xml_file))
