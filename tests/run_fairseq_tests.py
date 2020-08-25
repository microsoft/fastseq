import glob
import unittest
import sys
import os
import argparse

#Install required packages and set env
os.system('pip install editdistance')

#arguments
parser = argparse.ArgumentParser()
parser.add_argument("-fast", help="Use Fastseq ? ", type=bool, default=False)
parser.add_argument("-version", help="Fairseq version ", default='v0.9.0')
args = parser.parse_args()

if args.fast : 
    import fastseq
fairseq_branch  = args.version

#paths
fastseq_path = '/'.join(os.path.realpath(__file__).split('/')[0:-2]) 
fairseq_dir ='/tmp/' 
fairseq_path =fairseq_dir+'/fairseq/'

#cloning fairseq in /tmp/
os.system('export PYTHONPATH='+fairseq_path+':$PYTHONPATH') 
os.system ('rm -rf '+fairseq_path)
os.system('git clone --depth 1 --branch '+fairseq_branch+'  https://github.com/pytorch/fairseq.git '+fairseq_path) 


#Removing following path since it contains utils directory which clashes with utils.py file in fairseq/tests.
if fastseq_path in sys.path: 
    sys.path.remove(fastseq_path)
sys.path.remove(fastseq_path+'/tests')
if '' in sys.path:
    sys.path.remove('')
sys.path.insert(0, fairseq_path)



test_files = [ os.path.basename(x) for x in glob.glob(fairseq_path+'/tests/test_*.py')]
blocked_tests = ['test_binaries.py', 'test_bmuf.py', 'test_reproducibility.py']


for test in blocked_tests: 
    print('\n .... skipping ....'+test+'\n')
    

module_strings = ['tests.'+test_file[0:-3] for test_file in test_files if test_file not in blocked_tests]
suites = [unittest.defaultTestLoader.loadTestsFromName(test_file) for test_file in module_strings]
test_suite = unittest.TestSuite(suites)
test_runner = unittest.TextTestRunner().run(test_suite)
