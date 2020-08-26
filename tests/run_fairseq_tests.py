# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import glob
import unittest
import sys
import os
import argparse

#Install required packages and set env
os.system('pip install editdistance')

FASTSEQ_PATH = '/'.join(os.path.realpath(__file__).split('/')[0:-2])
FAIRSEQ_PATH = '/tmp/fairseq/'
FAIRSEQ_GIT_REPO = 'https://github.com/pytorch/fairseq.git'

def prepare_env (): 
    os.system('export PYTHONPATH=' + FAIRSEQ_PATH + ':$PYTHONPATH')
    #Removing following path since it contains utils directory 
    #which clashes with utils.py file in fairseq/tests.
    if FASTSEQ_PATH in sys.path:
        sys.path.remove(FASTSEQ_PATH)
    sys.path.remove(FASTSEQ_PATH + '/tests')
    if '' in sys.path:
        sys.path.remove('')
    sys.path.insert(0, FAIRSEQ_PATH)


def clone_fairseq (repo, version): 
    os.system('rm -rf ' + FAIRSEQ_PATH)
    os.system('git clone --depth 1 --branch ' + version +
          '  '+FAIRSEQ_GIT_REPO+' '+ FAIRSEQ_PATH)


def get_test_suites(test_files_path, blocked_tests) : 
    test_files = [
        os.path.basename(x) for x in glob.glob(test_files_path)
    ]
    
    for test in blocked_tests:
        print('\n .... skipping ....' + test + '\n')
    
    module_strings = [
        'tests.' + test_file[0:-3] for test_file in test_files
        if test_file not in blocked_tests
    ]
   
    suites = [
        unittest.defaultTestLoader.loadTestsFromName(test_file)
        for test_file in module_strings
    ]
    return suites
    

def run_test_suites ():
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--without_fastseq_opt", help="Run without Fastseq ", type=bool, default=False)
    parser.add_argument("--version", help="Fairseq version ", default='v0.9.0')
    parser.add_argument("--blocked_tests", nargs="+", default=['test_binaries.py', 'test_bmuf.py', 'test_reproducibility.py'])
    
    args = parser.parse_args()
    
    if not args.without_fastseq_opt:
        import fastseq #pylint: disable=import-outside-toplevel
    
    prepare_env()
    clone_fairseq(FAIRSEQ_GIT_REPO,args.version )

    test_files_path = FAIRSEQ_PATH + '/tests/test_*.py'
    suites = get_test_suites(test_files_path, args.blocked_tests)
    test_suite = unittest.TestSuite(suites)
    test_runner = unittest.TextTestRunner().run(test_suite)

if __name__ == "__main__":
    run_test_suites()
