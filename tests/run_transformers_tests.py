# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
""" script for importing transformers tests """

import io
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
TRANSFORMERS_PATH = '/tmp/transformers/'
TRANSFORMERS_GIT_URL = 'https://github.com/huggingface/transformers.git'

class TransformersUnitTests(parameterized.TestCase):
    """Run all the unit tests under transformers"""
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
        Repo.clone_from(repo,
                        TRANSFORMERS_PATH,
                        branch=version)
        pipmain(['install', '--editable', TRANSFORMERS_PATH])
        original_pythonpath = os.environ[
            'PYTHONPATH'] if 'PYTHONPATH' in os.environ else ''
        os.environ[
            'PYTHONPATH'] = TRANSFORMERS_PATH + ':' + original_pythonpath

    @parameterized.named_parameters({
        'testcase_name': 'Normal',
        'without_fastseq_opt': False,
        'transformers_version': 'v4.12.0',
        'blocked_tests': ['modeling_reformer',
                          'multigpu',
                          'HfApiEndpoints',
                          'HfApiPublicTest',
                          "test_gpt2_model_att_mask_past",
                          "test_gpt2_model_past"
        ]
    })
    def test_suites(self, without_fastseq_opt, transformers_version,
                    blocked_tests):
        """run test suites"""
        self.clone_and_build_transformers(TRANSFORMERS_GIT_URL,
                                          transformers_version)
        if not without_fastseq_opt:
            import fastseq  #pylint: disable=import-outside-toplevel
        import pytest #pylint: disable=import-outside-toplevel
        self.prepare_env()
        os.chdir(TRANSFORMERS_PATH)
        blocked_tests_string = (
            ' and '.join([' not '+ test for test in blocked_tests]))
        exit_code = pytest.main(
            ['-sv', '-k' + blocked_tests_string,  './tests/'])
        assert str(exit_code).strip() == 'ExitCode.OK'

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
        print(
            "Save the log of transformers unit tests into %s" % (log_xml_file))
